import logging
import re
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class StreamProcessRewarder:
    def __init__(self, smoothing_function=None, bleu_weights=(0.4, 0.3, 0.2, 0.1),
                 comet_model_path=None, comet_gpus=0):
        self.chencherry = smoothing_function or SmoothingFunction().method1
        # 仅匹配中文字符、英文字母及数字（自动忽略标点和空白符）
        self.tokenizer_pattern = re.compile(r"[a-zA-Z0-9\'-]+|[一-龥]")
        # 1/2/3/4-gram 权重，短序列时 3/4-gram 靠 smoothing 兜底
        self.bleu_weights = bleu_weights
        # COMET model (lazy-loaded)
        self.comet_model_path = comet_model_path
        self.comet_gpus = comet_gpus  # 0=CPU, 1=GPU; wmt22-comet-da 约需 2-4GB 显存
        self._comet_model = None

    @staticmethod
    def _patch_comet_xlmr_for_transformers5():
        """transformers>=4.5x 下 XLM-R return_dict=False 只返回 2 项，
        而 unbabel-comet 仍按 (last_hidden, pooler, hidden_states) 解包，
        会触发 ValueError: expected 3, got 2，导致 COMET 分数全 0。
        """
        from comet.encoders.xlmr import XLMREncoder

        if getattr(XLMREncoder, "_speechllm_tf5_patch", False):
            return

        def forward(self, input_ids, attention_mask, **kwargs):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_states = outputs.last_hidden_state
            all_layers = outputs.hidden_states
            return {
                "sentemb": last_hidden_states[:, 0, :],
                "wordemb": last_hidden_states,
                "all_layers": all_layers,
                "attention_mask": attention_mask,
            }

        XLMREncoder.forward = forward
        XLMREncoder._speechllm_tf5_patch = True
        logging.info("[COMET] patched XLMREncoder.forward for transformers return_dict compatibility")

    def _load_comet_model(self):
        """Lazy-load COMET model from checkpoint."""
        if self._comet_model is not None:
            return
        from comet import load_from_checkpoint
        self._patch_comet_xlmr_for_transformers5()
        self._comet_model = load_from_checkpoint(
            self.comet_model_path, reload_hparams=True, local_files_only=True
        )

    def _tokenize(self, text_list):
        """只提取纯净的单词和汉字，过滤 <W> 和标点"""
        tokens = []
        for text in text_list:
            if text == '<W>':
                continue
            clean_text = text.replace('<W>', ' ')
            tokens.extend(self.tokenizer_pattern.findall(clean_text))
        return tokens

    def compute_single_bleu(self, ref_tokens, gen_tokens):
        """计算单次 BLEU-2"""
        if not ref_tokens:
            return 0.0

        ref_words = self._tokenize(ref_tokens)
        gen_words = self._tokenize(gen_tokens)

        if not ref_words:
            return 0.0

        return sentence_bleu([ref_words], gen_words, weights=self.bleu_weights, smoothing_function=self.chencherry)

    # ================= 独立接口 =================

    def compute_local_bleu(self, gen_chunks, target_chunks, chunk_id):
        """局部过程 BLEU：自动截取前缀并使 Reference 向后寻找文本对齐"""
        gen_prefix_seq = gen_chunks[:chunk_id + 1]

        ref_idx = chunk_id
        while ref_idx < len(target_chunks) - 1 and target_chunks[ref_idx].strip() == "":
            ref_idx += 1

        tgt_prefix_seq = target_chunks[:ref_idx + 1]

        return self.compute_single_bleu(tgt_prefix_seq, gen_prefix_seq)

    def compute_global_bleu(self, gen_chunks, target_chunks):
        """全局 BLEU（如果仍有单独调用的需要，保留该接口）"""
        return self.compute_single_bleu(target_chunks, gen_chunks)

    def compute_process_reward(self, gen_chunks, target_chunks, chunk_id, alpha=0.5):
        """
        论文公式 (3): r_t(i) = (1-α)*BLEU(ŷ_t, y_t) + α*BLEU(ŷ_T, y_T)
        - chunk_id: 当前帧索引 t 对应的 chunk 索引
        - alpha: 全局 BLEU 权重
        """
        local_bleu = self.compute_local_bleu(gen_chunks, target_chunks, chunk_id)
        global_bleu = self.compute_global_bleu(gen_chunks, target_chunks)
        return (1 - alpha) * local_bleu + alpha * global_bleu

    # ================= COMET 相关接口 =================

    def compute_global_comet(self, source_text, gen_full_text, ref_full_text):
        """单样本 COMET 分数。

        Args:
            source_text: 源语言文本（原始音频转写）
            gen_full_text: 模型生成的完整翻译
            ref_full_text: 参考翻译

        Returns:
            float: COMET score, 失败时返回 0.0
        """
        if self._comet_model is None:
            self._load_comet_model()
        if not source_text or not gen_full_text or not ref_full_text:
            return 0.0
        try:
            prev_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            predict_kwargs = {
                "batch_size": 1,
                "gpus": self.comet_gpus,
                "progress_bar": False,
            }
            if self.comet_gpus > 0 and prev_device is not None:
                # 多卡可见时强制落到当前 process 的 GPU（如 rank3 -> cuda:3）
                predict_kwargs["devices"] = [int(prev_device)]
            result = self._comet_model.predict(
                [{"src": source_text, "mt": gen_full_text, "ref": ref_full_text}],
                **predict_kwargs,
            )
            if prev_device is not None:
                torch.cuda.set_device(prev_device)
            return float(result.scores[0])
        except Exception as e:
            logging.warning(f"[COMET] compute_global_comet failed: {type(e).__name__}: {e}")
            return 0.0

    def compute_global_comet_batch(self, comet_samples):
        """批量 COMET 分数。

        Args:
            comet_samples: List[Dict], 每个 dict 含 {"src", "mt", "ref"}

        Returns:
            List[float]: COMET scores, 与输入顺序一致
        """
        if self._comet_model is None:
            self._load_comet_model()
        if not comet_samples:
            return []
        try:
            prev_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            predict_kwargs = {
                "batch_size": len(comet_samples),
                "gpus": self.comet_gpus,
                "progress_bar": False,
            }
            if self.comet_gpus > 0 and prev_device is not None:
                # 多卡可见时强制落到当前 process 的 GPU（如 rank3 -> cuda:3）
                predict_kwargs["devices"] = [int(prev_device)]
            result = self._comet_model.predict(
                comet_samples,
                **predict_kwargs,
            )
            if prev_device is not None:
                torch.cuda.set_device(prev_device)
            return [float(s) for s in result.scores]
        except Exception as e:
            logging.warning(
                f"[COMET] compute_global_comet_batch failed "
                f"({len(comet_samples)} samples): {type(e).__name__}: {e}"
            )
            return [0.0] * len(comet_samples)

    def compute_process_reward_comet(self, gen_chunks, target_chunks, chunk_id,
                                      alpha=0.5, comet_score=None):
        """带 COMET 全局项的过程奖励。

        r_t(i) = (1-α) * local_bleu(ŷ_t, y_t) + α * global_comet
        """
        local_bleu = self.compute_local_bleu(gen_chunks, target_chunks, chunk_id)
        if comet_score is None:
            comet_score = 0.0
        return (1 - alpha) * local_bleu + alpha * comet_score


if __name__ == "__main__":
    rewarder = StreamProcessRewarder()

    gen_test = ["", "Im a pig，", "", "而且我很胖。"]
    tgt_test = ["", "I'm a pig!", "而且我非常胖！", ""]

    print("-> 局部计算 (chunk_id=1):")
    local_bleu = rewarder.compute_local_bleu(gen_test, tgt_test, chunk_id=2)
    print(f"Local BLEU: {local_bleu:.4f}\n")

    print("-> 全局奖励 (完整序列):")
    process_reward = rewarder.compute_global_bleu(gen_test, tgt_test)
    print(f"Process Reward: {process_reward:.4f}")
