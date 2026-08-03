import copy
import torch
import logging
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import requests

from .trainer_ast_stream import SpeechLLMLightningStreamAST
from .stream_process_rewarder import StreamProcessRewarder
from speechllm.utils.shm_transport import (
    write_tensor_to_shm,
    cleanup_shm_file,
    cleanup_all_shm,
)
        
class SpeechLLMLightningASTGRPO(SpeechLLMLightningStreamAST):
    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__(model_config, train_ds, val_ds)
        # GRPO 专属超参数
        self.G = model_config.train.get("group_size", 8)
        self.clip_eps = model_config.train.get("clip_eps", 0.2)
        self.alpha = model_config.train.get("alpha", 0.5)  # 论文公式(3) 中的 α
        # π_old (旧模型): 用于 KL 散度，0 表示永不更新（冻结）
        self.old_policy_sync_interval = model_config.train.get("old_policy_sync_interval", 0)
        # π_ref (参考模型): 用于 PPO ratio，定期从 actor 同步
        self.ref_model_sync_interval = model_config.train.get("ref_model_sync_interval", 4)
        self._steps_since_ref_sync = 0
        self._steps_since_old_sync = 0
        # KL 散度系数（0 表示不计算 KL 散度损失）
        self.kl_coef = model_config.train.get("kl_coef", 0.0)
        # 用普通 dict 存储，避免被 Lightning/DeepSpeed 注册为子模块
        # （nn.Module 属性会被自动追踪，DeepSpeed 保存时会因 frozen param 报错）
        self._non_module_store = {}  # keys: "old_model" (KL), "ref_model" (ratio)

        # 推理采样参数
        self.gen_temperature = model_config.train.get("temperature", 0.8)
        self.gen_top_p = model_config.train.get("top_p", 0.95)
        self.gen_max_new_tokens = model_config.train.get("max_new_tokens", 256)
        self.gen_repetition_penalty = model_config.train.get("repetition_penalty", 1.0)
            
        # 实例化我们上一步写的奖励器
        bleu_weights = tuple(model_config.train.get("bleu_weights", [0.4, 0.3, 0.2, 0.1]))
        self.comet_model_path = model_config.train.get("comet_model_path", None)
        self.use_comet = self.comet_model_path is not None
        self.comet_gpus = model_config.train.get("comet_gpus", 0)  # 0=CPU, 1=GPU
        # 仅在指定 rank 加载/运行 COMET；None 表示每张卡各自加载
        self.comet_rank = model_config.train.get("comet_rank", None)
        self.rewarder = StreamProcessRewarder(
            bleu_weights=bleu_weights,
            comet_model_path=self.comet_model_path,
            comet_gpus=self.comet_gpus,
        )
        self.sglang_server_url = model_config.train.get("sglang_server_url", "http://127.0.0.1:30000")
        self.debug = model_config.train.get("debug", False)

        # SGLang 权重同步配置
        self.sglang_sync_interval = model_config.train.get("sglang_sync_interval", 50)
        self.sglang_weight_sync_path = model_config.train.get("sglang_weight_sync_path", "/tmp/sglang_sync_weights")
        self._steps_since_sglang_sync = 0

        # 共享内存代理配置
        self.use_shm_proxy = model_config.train.get("use_shm_proxy", False)
        self.shm_proxy_port_offset = model_config.train.get("shm_proxy_port_offset", 100)
        self._shm_proxy_process = None  # subprocess.Popen handle
        self._shm_proxy_url = None  # 在 _launch_shm_proxy 中设置

        # 预训练权重在 setup(stage="fit") 中由基类统一加载，
        # 不在 __init__ 中重复调用，避免 DeepSpeed 初始化后二次加载导致权重未变化警告。
        # 对应地，Trainer 需设置 num_sanity_val_steps=0 跳过 sanity check。
        
    def split_audio_features_into_chunks(self, audio_feature_tensor, target_metadata, max_audio_len):
        """根据 target_metadata 中的 start_idx/end_idx 将音频特征切分为 chunk 列表"""
        chunks = []
        for seg in target_metadata:
            s_idx = max(0, min(seg['start_idx'], max_audio_len))
            e_idx = max(s_idx, min(seg['end_idx'], max_audio_len))
            chunks.append(audio_feature_tensor[s_idx:e_idx])
        return chunks
    
    def _sglang_generate(self, input_embeds: torch.Tensor, sampling_params: dict) -> str:
        """
        将完整 embedding tensor 发送到 SGLang server /generate 接口。
        """
        embeds_list = input_embeds.float().cpu().tolist()

        data = {
            "input_embeds": embeds_list,
            "sampling_params": sampling_params,
        }
        resp = requests.post(f"{self.sglang_server_url}/generate", json=data)

        resp.raise_for_status()
        result = resp.json()
        gen_text = result["text"]

        if self.debug:
            gen_ids = self.llm_tokenizer(gen_text, add_special_tokens=False).input_ids
            logging.warning(
                f"[SGLang] input_len={input_embeds.shape[0]} "
                f"output_len={len(gen_ids)} "
                f"output={repr(gen_text)}"
            )
        return gen_text

    @torch.no_grad()
    def generate_via_sglang(self, speech_embeds, speech_embeds_length, prompt_ids, prompt_lens, targets_metadata, G=8, temperature=None, top_p=None, batch_idx=0):
        """
        逐 chunk 推理，同一 chunk_idx 下所有 (batch, G) 的请求并行发送给 SGLang。
        chunk 之间仍然串行（chunk t+1 依赖 chunk t 的生成结果）。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if temperature is None:
            temperature = self.gen_temperature
        if top_p is None:
            top_p = self.gen_top_p
        batch_size = speech_embeds.shape[0]
        dtype = speech_embeds.dtype
        device = speech_embeds.device

        embedder = getattr(self.llm_model.model, "embed_tokens", None)
        if embedder is None:
            embedder = self.llm_model.model.model.embed_tokens

        if getattr(self, "finetune_special_tokens", False) or getattr(self, "use_lora", False):
            token_list = list(self.special_tokens_dict.keys())
            a_idx = token_list.index('<A>')
            a_end_idx = token_list.index('</A>')
            emb_A = self.special_token_input_patch[a_idx].unsqueeze(0).to(dtype=dtype, device=device)
            emb_A_end = self.special_token_input_patch[a_end_idx].unsqueeze(0).to(dtype=dtype, device=device)
        else:
            special_ids = torch.tensor([self.token_A_id, self.token_A_end_id], device=device)
            s_embs = embedder(special_ids).to(dtype=dtype)
            emb_A, emb_A_end = s_embs[0:1], s_embs[1:2]

        stop_token_str = self.llm_tokenizer.decode([self.token_W_id], skip_special_tokens=False)
        if self.debug:
            logging.warning(f"[SGLang] stop token id={self.token_W_id}, repr={repr(stop_token_str)}")

        sampling_params = {
            "temperature": temperature,
            "top_p": top_p if temperature > 0 else 1.0,
            "stop": [stop_token_str],
            "max_new_tokens": self.gen_max_new_tokens,
            "skip_special_tokens": False,
            "repetition_penalty": self.gen_repetition_penalty,
        }

        history_embeds = []
        text_results = []
        audio_chunks_all = []
        num_chunks_all = []

        for b in range(batch_size):
            p_len = prompt_lens[b]
            prompt_embed = embedder(prompt_ids[b][:p_len].to(device)).to(dtype=dtype)
            audio_chunks = self.split_audio_features_into_chunks(
                speech_embeds[b], targets_metadata[b], speech_embeds_length[b].item()
            )
            audio_chunks_all.append(audio_chunks)
            num_chunks_all.append(len(audio_chunks))

            b_history = []
            b_texts = []
            for g in range(G):
                b_history.append([prompt_embed])
                b_texts.append([])
            history_embeds.append(b_history)
            text_results.append(b_texts)

        max_chunks = max(num_chunks_all) if num_chunks_all else 0

        def _send_request(embeds_tensor, params, max_retries=3, retry_delay=5):
            import time
            if self.use_shm_proxy and self._shm_proxy_url:
                return _send_request_shm(embeds_tensor, params, max_retries, retry_delay)
            embeds_list = embeds_tensor.float().cpu().tolist()
            data = {
                "input_embeds": embeds_list,
                "sampling_params": params,
            }
            for attempt in range(max_retries):
                try:
                    resp = requests.post(f"{self.sglang_server_url}/generate", json=data, timeout=120)
                    resp.raise_for_status()
                    result = resp.json()
                    result["_input_len"] = embeds_tensor.shape[0]
                    return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"[SGLang] request failed (attempt {attempt+1}/{max_retries}): {e}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise

        def _send_request_shm(embeds_tensor, params, max_retries=3, retry_delay=5):
            """通过共享内存代理发送 embedding，避免训练进程做 .tolist()+JSON 序列化。"""
            import time as _time
            shm_path, shape, dtype_str = write_tensor_to_shm(
                embeds_tensor, dtype_str="float32"
            )
            req_data = {
                "shm_path": shm_path,
                "shape": shape,
                "dtype": dtype_str,
                "sampling_params": params,
            }
            for attempt in range(max_retries):
                try:
                    resp = requests.post(
                        f"{self._shm_proxy_url}/generate_shm",
                        json=req_data,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    result["_input_len"] = shape[0]
                    return result
                except Exception as e:
                    cleanup_shm_file(shm_path)
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"[SGLang-SHM] request failed "
                            f"(attempt {attempt + 1}/{max_retries}): {e}, "
                            f"retrying in {retry_delay}s..."
                        )
                        shm_path, shape, dtype_str = write_tensor_to_shm(
                            embeds_tensor, dtype_str="float32"
                        )
                        req_data["shm_path"] = shm_path
                        req_data["shape"] = shape
                        req_data["dtype"] = dtype_str
                        _time.sleep(retry_delay)
                    else:
                        raise

        for chunk_idx in range(max_chunks):
            futures = {}
            with ThreadPoolExecutor(max_workers=batch_size * G) as executor:
                for b in range(batch_size):
                    if chunk_idx >= num_chunks_all[b]:
                        continue
                    audio_slice = audio_chunks_all[b][chunk_idx]
                    chunk_audio_emb = torch.cat([emb_A, audio_slice, emb_A_end], dim=0)

                    for g in range(G):
                        history_embeds[b][g].append(chunk_audio_emb)
                        full_embeds = torch.cat(history_embeds[b][g], dim=0)

                        fut = executor.submit(_send_request, full_embeds, sampling_params)
                        futures[fut] = (b, g)

                for fut in as_completed(futures):
                    b, g = futures[fut]
                    result = fut.result()
                    gen_text = result["text"]

                    if self.debug:
                        gen_token_ids = self.llm_tokenizer(gen_text, add_special_tokens=False).input_ids
                        logging.warning(
                            f"[SGLang] gpu={device} b={batch_idx} s={b} g={g} chunk={chunk_idx} "
                            f"input_len={result.get('_input_len', '?')} "
                            f"output_len={len(gen_token_ids)} "
                            f"output={repr(gen_text[:80])}"
                        )

                    text_results[b][g].append(gen_text)

                    gen_ids = self.llm_tokenizer(
                        gen_text, return_tensors="pt", add_special_tokens=False
                    ).input_ids[0].to(device)
                    if gen_ids.numel() > 0:
                        gen_embed = embedder(gen_ids).to(dtype=dtype)
                        history_embeds[b][g].append(gen_embed)

        sglang_states_G = []
        for b in range(batch_size):
            for g in range(G):
                sglang_states_G.append(text_results[b][g])

        return sglang_states_G

    def _should_own_comet(self) -> bool:
        """当前 rank 是否负责加载/运行 COMET。"""
        if not self.use_comet:
            return False
        if self.comet_rank is None:
            return True
        return int(self.global_rank) == int(self.comet_rank)

    def _warmup_comet_if_needed(self) -> None:
        """仅在拥有 COMET 的 rank 上预热模型。"""
        if not self._should_own_comet():
            if self.use_comet and self.comet_rank is not None:
                logging.info(
                    f"[GPU {self.global_rank}] skip COMET load "
                    f"(owned by rank {self.comet_rank})"
                )
            return
        self.rewarder._load_comet_model()
        logging.info(
            f"[GPU {self.global_rank}] COMET model loaded from {self.comet_model_path}"
        )

    def _compute_comet_scores_distributed(self, comet_samples):
        """在 comet_rank 上集中计算 COMET，再把本 rank 的分数返回。

        comet_rank is None 时退化为本地计算。
        """
        if not self.use_comet:
            return None

        if (
            self.comet_rank is None
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return self.rewarder.compute_global_comet_batch(comet_samples)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        comet_rank = int(self.comet_rank)
        if not (0 <= comet_rank < world_size):
            raise ValueError(
                f"comet_rank={comet_rank} out of range for world_size={world_size}"
            )

        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, comet_samples)

        result_per_rank = None
        if rank == comet_rank:
            sizes = [len(samples or []) for samples in gathered]
            flat = []
            for samples in gathered:
                if samples:
                    flat.extend(samples)
            all_scores = (
                self.rewarder.compute_global_comet_batch(flat) if flat else []
            )
            result_per_rank = []
            offset = 0
            for sz in sizes:
                result_per_rank.append(all_scores[offset:offset + sz])
                offset += sz

        obj_list = [result_per_rank]
        torch.distributed.broadcast_object_list(obj_list, src=comet_rank)
        result_per_rank = obj_list[0]
        return result_per_rank[rank]

    def compute_process_bleu_rewards(self, sglang_states_G, targets_metadata, batch_size, batch_idx=None):
        """
        按论文公式 (3)(5)(6) 计算逐帧过程奖励与优势。
        当 self.use_comet 为 True 时，全局项使用 COMET 分数替代全局 BLEU。
        返回:
          normalized_rewards: list of list, shape [B*G][num_chunks], 每帧归一化奖励 r̄_t(i)
          advantages:         list of list, shape [B*G][num_chunks], 每帧优势 R_t(i) = Σ_{t'>t} r̄_t'(i)
          all_raw:            list of list, shape [B*G][num_chunks], 原始过程奖励
        """
        # 在计算奖励前，打印每个样本的参考分片文本和每个推理的分片文本
        for b in range(batch_size):
            target_chunks = [seg['text'] for seg in targets_metadata[b]]
            logging.warning(
                f"[GRPO-Reward] sample={b} ref_chunks({len(target_chunks)}): "
                + " | ".join([repr(c) for c in target_chunks])
            )
            for g in range(self.G):
                idx = b * self.G + g
                gen_chunks = sglang_states_G[idx]
                logging.warning(
                    f"[GRPO-Reward] batch={batch_idx} sample={b} g={g} gen_chunks({len(gen_chunks)}): "
                    + " | ".join([repr(c) for c in gen_chunks])
                )

        # 如果启用 COMET，预计算所有 B*G 个序列的全局 COMET 分数
        comet_scores = None
        if self.use_comet:
            comet_samples = []
            for b in range(batch_size):
                source_text = (
                    targets_metadata[b][0].get('source_text', '')
                    if targets_metadata[b] else ''
                )
                ref_full_text = ' '.join(
                    seg['text'] for seg in targets_metadata[b]
                )
                for g in range(self.G):
                    idx = b * self.G + g
                    gen_full_text = ' '.join(sglang_states_G[idx])
                    comet_samples.append({
                        "src": source_text,
                        "mt": gen_full_text,
                        "ref": ref_full_text,
                    })
            comet_scores = self._compute_comet_scores_distributed(comet_samples)
            self._last_comet_scores = comet_scores  # 保存供 training_step 日志使用

        # step1: 计算每个生成、每个 chunk 的原始过程奖励 r_t(i)，公式(3)
        # raw_rewards[b][g][chunk_id] = r_t(i)
        # 同时记录分解后的 local_bleu 和 global_reward 供日志使用
        all_raw = []  # shape: [B*G, num_chunks]
        local_bleu_means = []   # shape: [B*G], 每个生成的平均 local BLEU
        global_rewards = []     # shape: [B*G], 每个生成的全局奖励 (BLEU or COMET)
        for b in range(batch_size):
            target_chunks = [seg['text'] for seg in targets_metadata[b]]
            num_chunks = len(target_chunks)
            for g in range(self.G):
                idx = b * self.G + g
                gen_chunks = sglang_states_G[idx]
                if self.use_comet and comet_scores is not None:
                    rewards_per_frame = [
                        self.rewarder.compute_process_reward_comet(
                            gen_chunks, target_chunks, t, self.alpha,
                            comet_score=comet_scores[idx],
                        )
                        for t in range(num_chunks)
                    ]
                    global_rewards.append(comet_scores[idx])
                else:
                    rewards_per_frame = [
                        self.rewarder.compute_process_reward(gen_chunks, target_chunks, t, self.alpha)
                        for t in range(num_chunks)
                    ]
                    global_rewards.append(
                        self.rewarder.compute_global_bleu(gen_chunks, target_chunks)
                    )
                # 平均 local BLEU 跨所有 chunk
                if num_chunks > 0:
                    local_bleus = [
                        self.rewarder.compute_local_bleu(gen_chunks, target_chunks, t)
                        for t in range(num_chunks)
                    ]
                    local_bleu_means.append(sum(local_bleus) / num_chunks)
                else:
                    local_bleu_means.append(0.0)
                all_raw.append(rewards_per_frame)

        # 保存分解奖励供 training_step 日志使用
        self._last_local_bleu_means = local_bleu_means
        self._last_global_rewards = global_rewards

        # step2: 按帧索引跨 G 归一化，公式(5): r̄_t(i) = (r_t(i) - mean_k) / std_k
        # all_raw: [B*G, num_chunks] -> 对每个 b 的 G 个生成做归一化
        normalized = []
        for b in range(batch_size):
            group = all_raw[b * self.G : b * self.G + self.G]  # [G, num_chunks]
            num_chunks = len(group[0])
            norm_group = []
            for g in range(self.G):
                norm_group.append([0.0] * num_chunks)
            for t in range(num_chunks):
                vals = [group[g][t] for g in range(self.G)]
                mean_t = sum(vals) / self.G
                std_t = (sum((v - mean_t) ** 2 for v in vals) / self.G) ** 0.5 + 1e-8
                for g in range(self.G):
                    norm_group[g][t] = (group[g][t] - mean_t) / std_t
            normalized.extend(norm_group)

        # step3: 计算优势 R_t(i) = Σ_{t'>=t} r̄_t'(i)，公式(6)
        # 注意：包含当前帧 t 自身的奖励，否则最后一帧优势恒为 0
        advantages = []
        for bg in range(batch_size * self.G):
            nr = normalized[bg]
            adv = [sum(nr[t2] for t2 in range(t, len(nr))) for t in range(len(nr))]
            advantages.append(adv)

        return normalized, advantages, all_raw
    
    def prepare_rl_forward_inputs(self, speech_embeds, speech_embeds_length, sglang_states_G, targets_metadata, prompt_ids, prompt_lens):
        """
        【极其关键】：将 SGLang 生成的无梯度纯文本，重新与带梯度的 Audio Embeddings 组合。
        逻辑完美镜像 SFT 的借位法，保证生成 <W> 的行为得到正确训练。

        优化：将所有 prompt 和 segment 的 token ids 收集起来，一次性调用 embedder，
        避免循环内多次 CUDA kernel launch。
        """
        batch_size = speech_embeds.shape[0]
        dtype = speech_embeds.dtype
        device = self.device
        
        embedder = getattr(self.llm_model.model, "embed_tokens", None)
        if embedder is None:
            embedder = self.llm_model.model.model.embed_tokens

        # 获取特殊 Token 词向量（与 generate_via_sglang 保持一致）
        if getattr(self, "finetune_special_tokens", False) or getattr(self, "use_lora", False):
            token_list = list(self.special_tokens_dict.keys())
            a_idx = token_list.index('<A>')
            a_end_idx = token_list.index('</A>')
            emb_A = self.special_token_input_patch[a_idx].unsqueeze(0).to(dtype=dtype, device=device)
            emb_A_end = self.special_token_input_patch[a_end_idx].unsqueeze(0).to(dtype=dtype, device=device)
        else:
            special_ids = torch.tensor([self.token_A_id, self.token_A_end_id], device=device)
            s_embs = embedder(special_ids).to(dtype=dtype)
            emb_A, emb_A_end = s_embs[0:1], s_embs[1:2]
        embed_dim = embedder.weight.size(1) 

        # ====== 第一步：收集所有需要 embed 的 token ids ======
        all_id_chunks = []   # 每个元素是一个 1D tensor
        all_id_lens = []     # 对应长度

        # 收集 prompt ids，每个 batch 只需 embed 一次（G 个生成共享同一 prompt）
        prompt_chunk_indices = []
        for b in range(batch_size):
            p_len = prompt_lens[b].item() if isinstance(prompt_lens[b], torch.Tensor) else prompt_lens[b]
            p_ids = prompt_ids[b][:p_len].to(device)
            prompt_chunk_indices.append(len(all_id_chunks))
            all_id_chunks.append(p_ids)
            all_id_lens.append(p_len)

        # 收集 SGLang 生成文本的 token ids
        # seg_token_ids[b][g][seg_idx] = token ids tensor (可能为空)
        # seg_chunk_indices[b][g][seg_idx] = 在 all_id_chunks 中的索引（-1 表示空文本）
        seg_token_ids = []
        seg_chunk_indices = []
        for b in range(batch_size):
            b_ids = []
            b_indices = []
            for g in range(self.G):
                idx = b * self.G + g
                gen_text_chunks = sglang_states_G[idx]
                g_ids = []
                g_indices = []
                for gen_text in gen_text_chunks:
                    txt_ids = self.llm_tokenizer(
                        gen_text, return_tensors="pt", add_special_tokens=False
                    ).input_ids[0].to(device)
                    g_ids.append(txt_ids)
                    if txt_ids.numel() > 0:
                        g_indices.append(len(all_id_chunks))
                        all_id_chunks.append(txt_ids)
                        all_id_lens.append(txt_ids.numel())
                    else:
                        g_indices.append(-1)
                b_ids.append(g_ids)
                b_indices.append(g_indices)
            seg_token_ids.append(b_ids)
            seg_chunk_indices.append(b_indices)

        # 一次性 embed 所有 token ids
        if all_id_chunks:
            all_ids_cat = torch.cat(all_id_chunks, dim=0)
            all_embeds_flat = embedder(all_ids_cat).to(dtype=dtype)
            all_embeds_split = list(torch.split(all_embeds_flat, all_id_lens))
        else:
            all_embeds_split = []

        # ====== 第二步：组装序列（不再调用 embedder）======
        batch_seq_embeds = []
        batch_seq_labels = []
        
        for b in range(batch_size):
            full_audio = speech_embeds[b]
            max_audio_len = speech_embeds_length[b].item()
            curr_p_embed = all_embeds_split[prompt_chunk_indices[b]]
            curr_p_len = all_id_lens[prompt_chunk_indices[b]]
            
            for g in range(self.G):
                idx = b * self.G + g
                gen_text_chunks = sglang_states_G[idx]
                
                current_embeds = [curr_p_embed]
                current_labels = [torch.full((curr_p_len,), -100, dtype=torch.long, device=device)]

                for seg_idx, gen_text in enumerate(gen_text_chunks):
                    # 获取该 Chunk 对应的音频范围
                    seg_meta = targets_metadata[b][seg_idx]
                    s_idx = max(0, min(seg_meta['start_idx'], max_audio_len))
                    e_idx = max(s_idx, min(seg_meta['end_idx'], max_audio_len))
                    audio_slice = full_audio[s_idx:e_idx]

                    # 从预计算的 embeds 中取 text embedding
                    chunk_idx = seg_chunk_indices[b][g][seg_idx]
                    txt_ids = seg_token_ids[b][g][seg_idx]
                    if chunk_idx >= 0:
                        txt_embed = all_embeds_split[chunk_idx]
                    else:
                        txt_embed = torch.empty((0, embed_dim), device=device, dtype=dtype)
                    
                    chunk_input_emb = torch.cat([emb_A, audio_slice, emb_A_end], dim=0)
                    chunk_input_lab = torch.full((chunk_input_emb.size(0),), -100, dtype=torch.long, device=device)

                    # 【借位法复刻】
                    if seg_idx > 0:
                        chunk_input_lab[0] = self.token_W_id

                    chunk_target_emb = txt_embed
                    chunk_target_lab = txt_ids
                    
                    current_embeds.extend([chunk_input_emb, chunk_target_emb])
                    current_labels.extend([chunk_input_lab, chunk_target_lab])

                # 最后一个 Dummy Token 托底 <W>
                if len(gen_text_chunks) > 0:
                    dummy_emb = torch.zeros((1, embed_dim), dtype=dtype, device=device)
                    final_lab = torch.tensor([self.token_W_id], dtype=torch.long, device=device)
                    current_embeds.append(dummy_emb)
                    current_labels.append(final_lab)

                batch_seq_embeds.append(torch.cat(current_embeds, dim=0))
                batch_seq_labels.append(torch.cat(current_labels, dim=0))

                # 截断过长序列，防止 OOM
                if self.max_token_length is not None:
                    seq_len = batch_seq_embeds[-1].size(0)
                    if seq_len > self.max_token_length:
                        batch_seq_embeds[-1] = batch_seq_embeds[-1][:self.max_token_length]
                        batch_seq_labels[-1] = batch_seq_labels[-1][:self.max_token_length]

        inputs_embeds = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=0.0)
        labels = pad_sequence(batch_seq_labels, batch_first=True, padding_value=-100)

        seq_lens = torch.tensor([s.size(0) for s in batch_seq_embeds], device=device)
        max_len = inputs_embeds.size(1)
        attention_mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        
        return inputs_embeds, labels, attention_mask
    
    def on_train_start(self) -> None:
        """Override 父类的 on_train_start，跳过重复加载 checkpoint（已在 __init__ 中完成）"""
        self._resume_global_step = 0
        # 不再调用 self._load_from_deepspeed_dir，因为 __init__ 已加载
        if getattr(self, '_resume_global_step', 0):
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = self._resume_global_step
            self.trainer.fit_loop.epoch_loop.batch_progress.total.completed = self._resume_global_step
            logging.info(f"恢复 global_step 至: {self._resume_global_step}")
        if not self.finetune_encoder:
            self.audio_encoder.eval()

        # 预热 COMET 模型（避免首次调用时阻塞训练）
        self._warmup_comet_if_needed()

        # 启动共享内存代理（如果启用）
        if self.use_shm_proxy and self._shm_proxy_process is None:
            self._launch_shm_proxy()

    def on_train_end(self) -> None:
        """训练结束时清理共享内存代理。"""
        self._shutdown_shm_proxy()

    def _launch_shm_proxy(self):
        """启动共享内存代理进程，将昂贵的 .tolist()+JSON 序列化卸载到独立进程。"""
        import os
        import signal
        import subprocess
        import time as _time

        # 每个 rank 使用独立端口，基准 = shm_proxy_port_offset，加 global_rank 保证唯一
        proxy_port = self.shm_proxy_port_offset + self.global_rank
        self._shm_proxy_port = proxy_port
        self._shm_proxy_url = f"http://127.0.0.1:{proxy_port}"

        cmd = [
            "python", "-m", "speechllm.utils.shm_proxy",
            "--sglang-url", self.sglang_server_url,
            "--port", str(proxy_port),
        ]

        log_dir = os.path.join(
            getattr(self, "exp_dir", "/tmp"), "sglang_logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"shm_proxy_rank{self.global_rank}.log")

        logging.info(
            f"[GPU {self.global_rank}] Launching SHM proxy on port {proxy_port} "
            f"-> SGLang {self.sglang_server_url}: {' '.join(cmd)}"
        )
        log_fh = open(log_path, "w")
        self._shm_proxy_log_fh = log_fh
        self._shm_proxy_process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # 等待代理就绪
        url = f"{self._shm_proxy_url}/health"
        deadline = _time.time() + 30
        while _time.time() < deadline:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logging.info(
                        f"[GPU {self.global_rank}] SHM proxy ready at "
                        f"{self._shm_proxy_url}"
                    )
                    return
            except requests.ConnectionError:
                pass
            _time.sleep(0.5)

        raise RuntimeError(
            f"[GPU {self.global_rank}] SHM proxy did not start within 30s"
        )

    def _shutdown_shm_proxy(self):
        """关闭共享内存代理进程。"""
        import os
        import signal

        proc = self._shm_proxy_process
        if proc is None:
            return
        logging.info(
            f"[GPU {self.global_rank}] Shutting down SHM proxy (pid={proc.pid})"
        )
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except Exception as e:
                logging.warning(
                    f"[GPU {self.global_rank}] Error shutting down SHM proxy: {e}"
                )
        finally:
            self._shm_proxy_process = None
            if hasattr(self, "_shm_proxy_log_fh"):
                self._shm_proxy_log_fh.close()
        cleanup_all_shm()

    def _sync_ref_model(self):
        """同步 π_ref (参考模型) 从当前 actor 参数，用于 PPO ratio。

        - ref_model_sync_interval == 0：仅初始化一次，之后永不更新。
        - ref_model_sync_interval > 0：首次 deepcopy 初始化，之后每 τ 步 load_state_dict。
        """
        ref_model = self._non_module_store.get("ref_model", None)
        if ref_model is None:
            ref_model = copy.deepcopy(self.llm_model)
            ref_model.eval()
            ref_model.requires_grad_(False)
            self._non_module_store["ref_model"] = ref_model
            logging.info(f"[GPU {self.global_rank}] Initialized π_ref (reference model) via deepcopy from current actor.")
        elif self.ref_model_sync_interval > 0:
            ref_model.load_state_dict(self.llm_model.state_dict())
            logging.info(f"[GPU {self.global_rank}] Synced π_ref from current actor parameters.")

    def _sync_old_policy(self):
        """同步 π_old (旧模型) 从当前 actor 参数，用于 KL 散度。

        - old_policy_sync_interval == 0：仅初始化一次，之后永不更新（冻结）。
        - old_policy_sync_interval > 0：首次 deepcopy 初始化，之后每 τ 步 load_state_dict。
        """
        old_model = self._non_module_store.get("old_model", None)
        if old_model is None:
            old_model = copy.deepcopy(self.llm_model)
            old_model.eval()
            old_model.requires_grad_(False)
            self._non_module_store["old_model"] = old_model
            logging.info(f"[GPU {self.global_rank}] Initialized π_old via deepcopy from current actor.")
        elif self.old_policy_sync_interval > 0:
            old_model.load_state_dict(self.llm_model.state_dict())
            logging.info(f"[GPU {self.global_rank}] Synced π_old from current actor parameters.")

    def _sync_sglang_weights(self):
        """将当前 LLM 权重保存到磁盘，然后通知 SGLang server 重新加载。
        DeepSpeed ZeRO-2 下需要先聚合分片权重，再由 rank 0 写盘。
        注意：save_pretrained 只保存 llm_model 本体权重，special_token_input_patch
        和 special_token_output_patch 是独立 nn.Parameter，需要手动写入 embed_tokens
        和 lm_head 对应位置，否则 SGLang 加载后特殊 token（如 <W>）的权重是随机值，
        模型不会输出结束符。
        """
        try:
            import deepspeed
            # 同步前先让 server 释放 KV cache，降低显存峰值
            if self.global_rank == 0:
                try:
                    requests.post(f"{self.sglang_server_url}/flush_cache", timeout=30)
                except Exception:
                    pass
            # 聚合所有 rank 的权重分片到 rank 0
            with deepspeed.zero.GatheredParameters(
                list(self.llm_model.parameters()), modifier_rank=0
            ):
                if self.global_rank == 0:
                    # 在 save 之前，将 special token patch 写入 llm_model 的 embed_tokens 和 lm_head
                    if getattr(self, "finetune_special_tokens", False) or getattr(self, "use_lora", False):
                        special_tokens_list = list(self.special_tokens_dict.keys())
                        input_embeds_weight = self.llm_model.get_input_embeddings().weight
                        output_embeds_weight = self.llm_model.get_output_embeddings().weight
                        with torch.no_grad():
                            for i, token in enumerate(special_tokens_list):
                                tid = self.special_token_ids[token]
                                input_embeds_weight[tid].copy_(self.special_token_input_patch.data[i])
                                output_embeds_weight[tid].copy_(self.special_token_output_patch.data[i])
                        logging.info(f"[GPU {self.global_rank}] Applied special_token patches to embed_tokens/lm_head before saving")

                    self.llm_model.save_pretrained(self.sglang_weight_sync_path)
                    self.llm_tokenizer.save_pretrained(self.sglang_weight_sync_path)
                    resp = requests.post(
                        f"{self.sglang_server_url}/update_weights_from_disk",
                        json={"model_path": self.sglang_weight_sync_path},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    if result.get("success"):
                        logging.info(f"[GPU {self.global_rank}] SGLang weights synced from {self.sglang_weight_sync_path}")
                    else:
                        logging.warning(f"[GPU {self.global_rank}] SGLang weight sync failed: {result.get('message')}")
        except Exception as e:
            logging.warning(f"[GPU {self.global_rank}] SGLang weight sync error (non-fatal): {e}")

    def training_step(self, batch, batch_idx):
        batch_features, audio_lengths, targets_metadata, prompt_ids, prompt_lens = batch
        batch_size = batch_features.shape[0]

        self.total_processed_samples += batch_size
        self.epoch_processed_samples += batch_size

        # π_ref (参考模型) 同步 — 用于 PPO ratio，定期同步
        if (self.ref_model_sync_interval > 0 and self._steps_since_ref_sync >= self.ref_model_sync_interval) \
                or "ref_model" not in self._non_module_store:
            self._sync_ref_model()
            self._steps_since_ref_sync = 0
        self._steps_since_ref_sync += 1

        # π_old (旧模型) 同步 — 用于 KL 散度，old_policy_sync_interval=0 时仅初始化不更新
        if (self.old_policy_sync_interval > 0 and self._steps_since_old_sync >= self.old_policy_sync_interval) \
                or "old_model" not in self._non_module_store:
            self._sync_old_policy()
            self._steps_since_old_sync = 0
        self._steps_since_old_sync += 1

        need_sglang_sync = self._steps_since_sglang_sync >= self.sglang_sync_interval
        if need_sglang_sync:
            self._steps_since_sglang_sync = 0
        self._steps_since_sglang_sync += 1

        # 阶段 1: 推理 (SGLang)
        speech_embeds, speech_embeds_length = self.get_audio_embeds(batch_features, audio_lengths)

        with torch.no_grad():
            sglang_states_G = self.generate_via_sglang(
                speech_embeds, speech_embeds_length, prompt_ids, prompt_lens, targets_metadata, G=self.G, batch_idx=batch_idx
            )

        if need_sglang_sync:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self._sync_sglang_weights()

        with torch.no_grad():
            generated_inputs, generated_labels, attention_mask = self.prepare_rl_forward_inputs(
                speech_embeds, speech_embeds_length, sglang_states_G, targets_metadata, prompt_ids, prompt_lens
            )

        # 阶段 2: 计算逐帧过程奖励与优势
        normalized_rewards, advantages_per_frame, raw_rewards = self.compute_process_bleu_rewards(sglang_states_G, targets_metadata, batch_size, batch_idx)

        # 阶段 3: Actor + Ref Model (ratio) + Old Model (KL) 前向
        outputs = self.llm_model(inputs_embeds=generated_inputs, attention_mask=attention_mask, labels=generated_labels)
        with torch.no_grad():
            ref_outputs = self._non_module_store["ref_model"](inputs_embeds=generated_inputs, attention_mask=attention_mask, labels=generated_labels)
        old_logits = None
        if self.kl_coef > 0:
            with torch.no_grad():
                old_outputs = self._non_module_store["old_model"](inputs_embeds=generated_inputs, attention_mask=attention_mask, labels=generated_labels)
            old_logits = old_outputs.logits

        # 阶段 4: GRPO Loss
        policy_loss, kl_loss = self._compute_grpo_loss(
            outputs.logits, ref_outputs.logits, old_logits,
            generated_labels, targets_metadata, advantages_per_frame, batch_size,
        )
        total_loss = policy_loss + self.kl_coef * kl_loss

        mean_adv = sum(sum(a) for a in advantages_per_frame) / max(sum(len(a) for a in advantages_per_frame), 1)

        group_mean_rewards = []
        group_var_rewards = []
        for b in range(batch_size):
            group_rewards = []
            for g in range(self.G):
                idx = b * self.G + g
                nr = raw_rewards[idx]
                if nr:
                    group_rewards.append(sum(nr) / len(nr))
                else:
                    group_rewards.append(0.0)
            g_mean = sum(group_rewards) / self.G
            g_var = sum((r - g_mean) ** 2 for r in group_rewards) / self.G
            group_mean_rewards.append(g_mean)
            group_var_rewards.append(g_var)
            if self.debug:
                logging.warning(
                    f"[GRPO] b={batch_idx} s={b} "
                    f"group_mean_reward={g_mean:.4f} group_var={g_var:.4f} "
                    f"per_g={[f'{r:.3f}' for r in group_rewards]}"
                )

        avg_group_reward = sum(group_mean_rewards) / max(len(group_mean_rewards), 1)
        avg_group_var = sum(group_var_rewards) / max(len(group_var_rewards), 1)

        # 分解奖励：local BLEU 和 global reward 的均值
        local_bleu_means = getattr(self, '_last_local_bleu_means', None)
        global_rewards = getattr(self, '_last_global_rewards', None)
        avg_local_bleu = sum(local_bleu_means) / max(len(local_bleu_means), 1) if local_bleu_means else 0.0
        avg_global_reward = sum(global_rewards) / max(len(global_rewards), 1) if global_rewards else 0.0

        log_dict = {
            "rl/policy_loss": policy_loss,
            "rl/kl_loss": kl_loss,
            "rl/total_loss": total_loss,
            "rl/mean_advantage": mean_adv,
            "rl/group_mean_reward": avg_group_reward,
            "rl/group_reward_var": avg_group_var,
            "rl/local_bleu_mean": avg_local_bleu,
            "rl/global_reward_mean": avg_global_reward,
        }
        if self.use_comet:
            comet_scores = getattr(self, '_last_comet_scores', None)
            if comet_scores:
                log_dict["rl/comet_global_mean"] = sum(comet_scores) / len(comet_scores)
        self.log_dict(log_dict, prog_bar=True)

        # 打印分解奖励到日志
        global_label = "COMET" if self.use_comet else "BLEU"
        logging.warning(
            f"[GRPO-Reward] batch={batch_idx} "
            f"avg_local_bleu={avg_local_bleu:.4f} "
            f"avg_global_{global_label.lower()}={avg_global_reward:.4f} "
            f"avg_combined={avg_group_reward:.4f}"
        )
        self.log("total_samples", float(self.total_processed_samples), on_step=True, on_epoch=False, prog_bar=False)
        if self.sampler_type not in ("stateless_sampler", "shuffle_queue_stateless_sampler", "shar_stateless_sampler", "shard_pool_sampler"):
            self.log("epoch_samples", float(self.epoch_processed_samples), on_step=True, on_epoch=False, prog_bar=False)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_features, audio_lengths, targets_metadata, prompt_ids, prompt_lens = batch
        batch_size = batch_features.shape[0]

        with torch.no_grad():
            speech_embeds, speech_embeds_length = self.get_audio_embeds(batch_features, audio_lengths)
            sglang_states_1 = self.generate_via_sglang(
                speech_embeds, speech_embeds_length, prompt_ids, prompt_lens, targets_metadata, G=1, temperature=0.0, top_p=1.0
            )

        raw_rewards = []
        comet_scores = []
        local_bleus = []
        global_rews = []

        # 第一遍：计算 local BLEU，收集 COMET 输入
        all_target_chunks = []
        all_gen_chunks = []
        comet_samples = []
        for b in range(batch_size):
            target_chunks = [seg['text'] for seg in targets_metadata[b]]
            gen_chunks = sglang_states_1[b]
            all_target_chunks.append(target_chunks)
            all_gen_chunks.append(gen_chunks)
            logging.warning(
                f"[GRPO-Reward] val batch={batch_idx} sample={b} ref_chunks({len(target_chunks)}): "
                + " | ".join([repr(c) for c in target_chunks])
            )
            logging.warning(
                f"[GRPO-Reward] val batch={batch_idx} sample={b} gen_chunks({len(gen_chunks)}): "
                + " | ".join([repr(c) for c in gen_chunks])
            )
            num_chunks = len(target_chunks)

            # 计算 local BLEU（所有 chunk 的平均）
            local_bleu = 0.0
            if num_chunks > 0:
                local_bleu = sum(
                    self.rewarder.compute_local_bleu(gen_chunks, target_chunks, t)
                    for t in range(num_chunks)
                ) / num_chunks
            local_bleus.append(local_bleu)

            if self.use_comet:
                source_text = (
                    targets_metadata[b][0].get('source_text', '')
                    if targets_metadata[b] else ''
                )
                ref_full = ' '.join(target_chunks)
                gen_full = ' '.join(gen_chunks)
                comet_samples.append({"src": source_text, "mt": gen_full, "ref": ref_full})
            else:
                global_bleu = self.rewarder.compute_global_bleu(gen_chunks, target_chunks)
                global_rews.append(global_bleu)

        # 批量计算 COMET（comet_rank 模式下所有 rank 都必须进入 collective）
        if self.use_comet:
            comet_scores = self._compute_comet_scores_distributed(comet_samples)
            global_rews = list(comet_scores) if comet_scores else []

        # 第二遍：计算过程奖励
        for b in range(batch_size):
            target_chunks = all_target_chunks[b]
            gen_chunks = all_gen_chunks[b]
            num_chunks = len(target_chunks)

            if num_chunks > 0:
                if self.use_comet and comet_scores:
                    reward = sum(
                        self.rewarder.compute_process_reward_comet(
                            gen_chunks, target_chunks, t, self.alpha,
                            comet_score=comet_scores[b],
                        )
                        for t in range(num_chunks)
                    ) / num_chunks
                else:
                    reward = sum(
                        self.rewarder.compute_process_reward(gen_chunks, target_chunks, t, self.alpha)
                        for t in range(num_chunks)
                    ) / num_chunks
            else:
                reward = 0.0
            raw_rewards.append(reward)

        val_reward = torch.tensor(raw_rewards, dtype=torch.float, device=self.device).mean()
        val_local_bleu = torch.tensor(local_bleus, dtype=torch.float, device=self.device).mean()
        val_global_reward = torch.tensor(global_rews, dtype=torch.float, device=self.device).mean()

        global_label = "comet" if self.use_comet else "bleu"
        self.log_dict({
            "val/reward": val_reward,
            "val_reward_bleu": val_reward,  # 兼容旧 ModelCheckpoint monitor key
            "val/local_bleu": val_local_bleu,
            f"val/global_{global_label}": val_global_reward,
        }, prog_bar=True, sync_dist=True, batch_size=batch_size)
        if self.use_comet and comet_scores:
            val_comet = torch.tensor(comet_scores, dtype=torch.float, device=self.device).mean()
            self.log("val/comet", val_comet, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return val_reward

    def _compute_grpo_loss(self, logits, ref_logits, old_logits, labels, targets_metadata, advantages_per_frame, batch_size):
        """
        PPO clipped surrogate objective + KL 散度正则项。

        - ratio = π_θ / π_ref   (π_ref 用于 policy loss，定期同步)
        - KL = D_KL(π_θ || π_old)  (π_old 用于 KL 散度，冻结不更新)

        Returns:
            (policy_loss, kl_loss) 元组。
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()  # (B*G, L-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

        # 构建逐 token 的优势张量，按 chunk 对齐
        adv_tensor = self._build_chunk_advantage_tensor(shift_labels, targets_metadata, advantages_per_frame, batch_size)

        loss_mask = shift_labels != -100
        selected_lp = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
        selected_ref_lp = torch.gather(ref_log_probs, -1, shift_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)

        # PPO clipped surrogate: ratio = π_θ / π_ref
        ratio = torch.exp(selected_lp - selected_ref_lp)
        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_tensor
        loss = -torch.min(surr1, surr2)
        policy_loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        # KL 散度正则项: D_KL(π_θ || π_old)
        # 使用 k3 estimator (Schulman et al. 2020):
        #   kl = exp(log_old - log_θ) - (log_old - log_θ) - 1
        if self.kl_coef > 0 and old_logits is not None:
            shift_old_logits = old_logits[..., :-1, :].contiguous()
            old_log_probs = F.log_softmax(shift_old_logits, dim=-1)
            selected_old_lp = torch.gather(
                old_log_probs, -1,
                shift_labels.unsqueeze(-1).clamp(min=0)
            ).squeeze(-1)
            log_ratio_kl = selected_old_lp - selected_lp
            kl_per_token = torch.exp(log_ratio_kl) - log_ratio_kl - 1
            kl_loss = (kl_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        else:
            kl_loss = torch.tensor(0.0, device=loss_mask.device)

        return policy_loss, kl_loss

    def _build_chunk_advantage_tensor(self, shift_labels, targets_metadata, advantages_per_frame, batch_size):
        """
        将 advantages_per_frame[b*G+g][chunk_id] 映射到 shift_labels 中对应 chunk 的 token 位置。
        
        shift_labels 中非 -100 的连续段结构为：
          段0: [Text0_ids]                    -> chunk 0
          段1: [借位<W>]                       -> chunk 0 的结束（单 token）
          段2: [Text1_ids]                    -> chunk 1
          段3: [借位<W>]                       -> chunk 1 的结束
          ...
          最后: [末尾<W>]                      -> 最后一个 chunk 的结束
        
        借位的 <W> 是单 token 段（长度=1 且值=token_W_id），语义上属于前一个 chunk，
        不应触发 chunk_id 递增。只有长度 > 1 或非 <W> 的段才是新 chunk 的开始。
        """
        device = shift_labels.device
        adv_tensor = torch.zeros_like(shift_labels, dtype=torch.float)
        BG = batch_size * self.G
        for bg in range(BG):
            row = shift_labels[bg]  # (L-1,)
            adv_row = advantages_per_frame[bg]
            
            # 先找出所有非 -100 的连续段
            segments = []  # list of (start, end) 左闭右开
            in_seg = False
            seg_start = 0
            for pos in range(row.size(0)):
                if row[pos] != -100:
                    if not in_seg:
                        in_seg = True
                        seg_start = pos
                else:
                    if in_seg:
                        segments.append((seg_start, pos))
                        in_seg = False
            if in_seg:
                segments.append((seg_start, row.size(0)))
            
            # 将段分配到 chunk：
            # - 长度 == 1 且值 == token_W_id 的段是借位 <W>，属于前一个 chunk
            # - 其他段是新 chunk 的文本 token
            chunk_id = -1  # 还没遇到第一个文本段
            for seg_start, seg_end in segments:
                seg_len = seg_end - seg_start
                is_borrow_w = (seg_len == 1 and row[seg_start].item() == self.token_W_id)
                
                if not is_borrow_w:
                    # 新的文本段 -> 新 chunk
                    chunk_id += 1
                # else: 借位 <W> -> 仍属于当前 chunk_id
                
                if 0 <= chunk_id < len(adv_row):
                    for pos in range(seg_start, seg_end):
                        adv_tensor[bg, pos] = adv_row[chunk_id]
        
        return adv_tensor

    # --- 数学与张量对齐辅助函数 ---
    def gather_log_probs(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = shift_labels != -100
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)
        
        return selected_log_probs[loss_mask]
