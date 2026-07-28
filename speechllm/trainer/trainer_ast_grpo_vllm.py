"""
使用 vLLM 推理后端的 GRPO 训练器。

架构说明：
  - 每个训练 rank 以 Ray actor 的形式启动一个独立的 vLLM LLM 实例。
  - Ray actor 从训练进程接收多模态输入（音频 embedding + prompt），并返回生成文本。
  - 权重同步：rank 0 将 LLM 权重保存到磁盘（或共享内存），
    各 Ray actor 从磁盘重新加载。

与 SGLang 版本的主要区别：
  - 无 HTTP server，通信通过 Ray remote call 完成，无序列化开销。
  - vLLM 以"embedding 直接输入"模式工作，通过 TokensPrompt(prompt_embeds=...) 传入。
  - 每个训练 rank 独占一个 vLLM actor（默认 tp_size=1）。
"""

import gc
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from speechllm.trainer.trainer_ast_grpo import SpeechLLMLightningASTGRPO
from speechllm.utils.vllm_actor_cls import _get_vllm_actor_cls  # noqa: F401 — re-exported for back-compat


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SpeechLLMLightningASTGRPOVLLM(SpeechLLMLightningASTGRPO):
    """使用 vLLM Ray actor 进行推理的 GRPO 训练器。

    每个训练 rank 独占一个 Ray vLLM actor，部署在指定 GPU
    （actor_gpu_offset + local_rank）上。embedding 通过 Ray 对象存储传输，
    无 HTTP 序列化开销。

    配置项（位于 ``train`` 下）：
        vllm_model_path          : 初始化 vLLM 使用的 HF 模型路径
                                   （默认取 sglang_weight_sync_path）
        vllm_gpu_offset          : vLLM actor 起始 GPU 编号（默认 0）
        vllm_tp_size             : 每个 actor 的 tensor parallel size（默认 1）
        vllm_max_model_len       : vLLM 最大序列长度（默认 4096）
        vllm_gpu_memory_util     : vLLM 显存占用比例（默认 0.85）
        vllm_weight_sync_path    : 训练侧保存权重供 vLLM 重载的路径
                                   （默认取 sglang_weight_sync_path）
        vllm_sync_interval       : 权重同步间隔步数（默认 50）
        vllm_extra_kwargs        : 透传给 vLLM LLM() 的额外参数（dict）
        ray_address              : Ray 集群地址，如 "auto" 或 None
                                   （默认 None，即在本地启动集群）
    """

    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__(model_config, train_ds, val_ds)

        tcfg = model_config.train

        self.vllm_model_path = tcfg.get(
            "vllm_model_path",
            tcfg.get("sglang_weight_sync_path", "/tmp/vllm_init_weights"),
        )
        self.vllm_gpu_offset = tcfg.get("vllm_gpu_offset", 0)
        self.vllm_tp_size = tcfg.get("vllm_tp_size", 1)
        self.vllm_max_model_len = tcfg.get("vllm_max_model_len", 4096)
        self.vllm_gpu_memory_util = tcfg.get("vllm_gpu_memory_util", 0.85)
        self.vllm_weight_sync_path = tcfg.get(
            "vllm_weight_sync_path",
            tcfg.get("sglang_weight_sync_path", "/tmp/vllm_sync_weights"),
        )
        self.vllm_sync_interval = tcfg.get("vllm_sync_interval",
                                            tcfg.get("sglang_sync_interval", 50))
        self.vllm_extra_kwargs = tcfg.get("vllm_extra_kwargs", {})
        self.ray_address = tcfg.get("ray_address", None)

        # 替代基类的 SGLang 同步计数器，不使用 HTTP 路径
        self._steps_since_vllm_sync = 0
        # vLLM actor 句柄（ray.ObjectRef）
        self._vllm_actor = None

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        super_hook = getattr(super(), "on_fit_start", None)
        if super_hook:
            super_hook()
        self._init_ray()
        self._launch_vllm_actor()

    def on_train_start(self) -> None:
        # 跳过基类 on_train_start 中 SGLang 专属的 shm proxy 启动逻辑
        self._resume_global_step = 0
        if not self.finetune_encoder:
            self.audio_encoder.eval()
        # 确保 vLLM 初始权重目录存在
        self._save_llm_weights_for_vllm(self.vllm_weight_sync_path)

    def on_train_end(self) -> None:
        self._shutdown_vllm_actor()

    # ------------------------------------------------------------------
    # Ray / vLLM actor 管理
    # ------------------------------------------------------------------

    def _init_ray(self):
        import ray
        if not ray.is_initialized():
            ray.init(address=self.ray_address, ignore_reinit_error=True, namespace="speechllm_vllm")
            logging.info(
                f"[GPU {self.global_rank}] Ray 已初始化 "
                f"(address={self.ray_address})"
            )

    def _launch_vllm_actor(self):
        """为当前 rank 创建 Ray vLLM actor。"""
        import ray
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        gpu_id = self.vllm_gpu_offset + local_rank

        # rank 0 先写盘，确保 vLLM 有合法的 HF 模型目录可加载
        if self.global_rank == 0:
            self._save_llm_weights_for_vllm(self.vllm_model_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        VLLMEmbedActor = _get_vllm_actor_cls()
        self._vllm_actor = VLLMEmbedActor.remote(
            model_path=self.vllm_model_path,
            gpu_id=gpu_id,
            tp_size=self.vllm_tp_size,
            max_model_len=self.vllm_max_model_len,
            gpu_memory_utilization=self.vllm_gpu_memory_util,
            extra_kwargs=self.vllm_extra_kwargs,
        )

        # 轮询等待 actor 就绪（最多 300 秒）
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                ready = ray.get(self._vllm_actor.health.remote(), timeout=10)
                if ready:
                    logging.info(
                        f"[GPU {self.global_rank}] vLLM actor 就绪 "
                        f"(gpu={gpu_id})"
                    )
                    return
            except Exception as e:
                logging.debug(
                    f"[GPU {self.global_rank}] vLLM actor health 检查: {e}"
                )
            time.sleep(5)
        raise RuntimeError(
            f"[GPU {self.global_rank}] vLLM actor 在 300s 内未就绪"
        )

    def _shutdown_vllm_actor(self):
        if self._vllm_actor is None:
            return
        import ray
        try:
            ray.kill(self._vllm_actor)
            logging.info(f"[GPU {self.global_rank}] vLLM actor 已关闭")
        except Exception as e:
            logging.warning(f"[GPU {self.global_rank}] vLLM actor 关闭失败: {e}")
        self._vllm_actor = None

    # ------------------------------------------------------------------
    # 权重同步
    # ------------------------------------------------------------------

    def _save_llm_weights_for_vllm(self, save_path: str):
        """聚合 ZeRO-2 分片到 rank 0 并保存 HF 格式权重。"""
        try:
            import deepspeed
            with deepspeed.zero.GatheredParameters(
                list(self.llm_model.parameters()), modifier_rank=0
            ):
                if self.global_rank == 0:
                    # 将 special token patch 写入 embed_tokens 和 lm_head
                    if getattr(self, "finetune_special_tokens", False) or \
                            getattr(self, "use_lora", False):
                        special_tokens_list = list(self.special_tokens_dict.keys())
                        in_w = self.llm_model.get_input_embeddings().weight
                        out_w = self.llm_model.get_output_embeddings().weight
                        with torch.no_grad():
                            for i, token in enumerate(special_tokens_list):
                                tid = self.special_token_ids[token]
                                in_w[tid].copy_(self.special_token_input_patch.data[i])
                                out_w[tid].copy_(self.special_token_output_patch.data[i])
                    os.makedirs(save_path, exist_ok=True)
                    self.llm_model.save_pretrained(save_path)
                    self.llm_tokenizer.save_pretrained(save_path)
                    logging.info(f"[GPU 0] LLM 权重已保存至 {save_path}")
        except Exception as e:
            logging.warning(
                f"[GPU {self.global_rank}] _save_llm_weights_for_vllm 出错: {e}"
            )

    def _sync_vllm_weights(self):
        """保存权重后通知 vLLM actor 从磁盘重新加载。"""
        import ray
        try:
            self._save_llm_weights_for_vllm(self.vllm_weight_sync_path)

            # 等待 rank 0 写盘完成后再触发 actor 加载
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if self._vllm_actor is not None:
                ok = ray.get(
                    self._vllm_actor.reload_weights.remote(self.vllm_weight_sync_path),
                    timeout=300,
                )
                if ok:
                    logging.info(
                        f"[GPU {self.global_rank}] vLLM 权重已从 "
                        f"{self.vllm_weight_sync_path} 重新加载"
                    )
                else:
                    logging.warning(
                        f"[GPU {self.global_rank}] vLLM reload_weights 返回 False"
                    )
        except Exception as e:
            logging.warning(
                f"[GPU {self.global_rank}] vLLM 权重同步出错（非致命）: {e}"
            )

    # ------------------------------------------------------------------
    # 通过 Ray vLLM actor 推理（替代 generate_via_sglang）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_via_vllm(self, speech_embeds, speech_embeds_length,
                          prompt_ids, prompt_lens, targets_metadata,
                          G=8, temperature=None, top_p=None, batch_idx=0):
        """将 embedding 发送到 vLLM Ray actor 并返回生成文本。

        接口与 generate_via_sglang 保持一致：
          - chunk 间串行（chunk t+1 依赖 chunk t 的生成结果）
          - 同一 chunk 内所有 (batch, G) 请求并发发送给 actor

        返回：
            sglang_states_G: List[List[str]]，形状 [batch_size * G][num_chunks]
        """
        import ray

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

        if getattr(self, "finetune_special_tokens", False) or \
                getattr(self, "use_lora", False):
            token_list = list(self.special_tokens_dict.keys())
            a_idx = token_list.index('<A>')
            a_end_idx = token_list.index('</A>')
            emb_A = self.special_token_input_patch[a_idx].unsqueeze(0).to(dtype=dtype, device=device)
            emb_A_end = self.special_token_input_patch[a_end_idx].unsqueeze(0).to(dtype=dtype, device=device)
        else:
            special_ids = torch.tensor([self.token_A_id, self.token_A_end_id], device=device)
            s_embs = embedder(special_ids).to(dtype=dtype)
            emb_A, emb_A_end = s_embs[0:1], s_embs[1:2]

        sampling_params_dict = {
            "temperature": temperature,
            "top_p": top_p if temperature > 0 else 1.0,
            "stop_token_ids": [self.token_W_id],
            "max_tokens": self.gen_max_new_tokens,
            "skip_special_tokens": False,
            "repetition_penalty": self.gen_repetition_penalty,
        }

        # 初始化每个 (batch, G) 的历史 embedding 和输出文本累积器
        audio_chunks_all = []
        num_chunks_all = []
        history_embeds = []
        text_results = []

        for b in range(batch_size):
            p_len = prompt_lens[b]
            prompt_embed = embedder(prompt_ids[b][:p_len].to(device)).to(dtype=dtype)
            audio_chunks = self.split_audio_features_into_chunks(
                speech_embeds[b], targets_metadata[b], speech_embeds_length[b].item()
            )
            audio_chunks_all.append(audio_chunks)
            num_chunks_all.append(len(audio_chunks))

            b_history, b_texts = [], []
            for g in range(G):
                b_history.append([prompt_embed])
                b_texts.append([])
            history_embeds.append(b_history)
            text_results.append(b_texts)

        max_chunks = max(num_chunks_all) if num_chunks_all else 0

        def _ray_generate(embeds_tensor):
            """将单条 embedding 发送给 actor，阻塞等待结果返回。"""
            embeds_list = embeds_tensor.float().cpu().tolist()
            ref = self._vllm_actor.generate.remote(embeds_list, sampling_params_dict)
            return ray.get(ref, timeout=120)

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
                        fut = executor.submit(_ray_generate, full_embeds)
                        futures[fut] = (b, g)

                for fut in as_completed(futures):
                    b, g = futures[fut]
                    gen_text = fut.result()

                    text_results[b][g].append(gen_text)

                    # 将生成文本转回 embedding，拼入历史供下一 chunk 使用
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

    # ------------------------------------------------------------------
    # 覆盖 training_step 和 validation_step，使用 vLLM 推理
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch_features, audio_lengths, targets_metadata, prompt_ids, prompt_lens = batch
        batch_size = batch_features.shape[0]

        self.total_processed_samples += batch_size
        self.epoch_processed_samples += batch_size

        # π_ref (参考模型) 同步 — 用于 PPO ratio，定期同步
        if (self.ref_model_sync_interval > 0 and self._steps_since_ref_sync >= self.ref_model_sync_interval or
                "ref_model" not in self._non_module_store):
            self._sync_ref_model()
            self._steps_since_ref_sync = 0
        self._steps_since_ref_sync += 1

        # π_old (旧模型) 同步 — 用于 KL 散度，old_policy_sync_interval=0 时仅初始化不更新
        if (self.old_policy_sync_interval > 0 and self._steps_since_old_sync >= self.old_policy_sync_interval or
                "old_model" not in self._non_module_store):
            self._sync_old_policy()
            self._steps_since_old_sync = 0
        self._steps_since_old_sync += 1

        need_vllm_sync = self._steps_since_vllm_sync >= self.vllm_sync_interval
        if need_vllm_sync:
            self._steps_since_vllm_sync = 0
        self._steps_since_vllm_sync += 1

        # 阶段 1：音频编码 + vLLM 推理
        speech_embeds, speech_embeds_length = self.get_audio_embeds(
            batch_features, audio_lengths
        )

        with torch.no_grad():
            sglang_states_G = self.generate_via_vllm(
                speech_embeds, speech_embeds_length,
                prompt_ids, prompt_lens, targets_metadata,
                G=self.G, batch_idx=batch_idx,
            )

        if need_vllm_sync:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self._sync_vllm_weights()

        with torch.no_grad():
            generated_inputs, generated_labels, attention_mask = \
                self.prepare_rl_forward_inputs(
                    speech_embeds, speech_embeds_length,
                    sglang_states_G, targets_metadata,
                    prompt_ids, prompt_lens,
                )

        # 阶段 2：计算逐帧过程奖励与优势
        normalized_rewards, advantages_per_frame, raw_rewards = \
            self.compute_process_bleu_rewards(
                sglang_states_G, targets_metadata, batch_size
            )

        # 阶段 3：actor + ref model (ratio) + old model (KL) 前向
        outputs = self.llm_model(
            inputs_embeds=generated_inputs,
            attention_mask=attention_mask,
            labels=generated_labels,
        )
        with torch.no_grad():
            ref_outputs = self._non_module_store["ref_model"](
                inputs_embeds=generated_inputs,
                attention_mask=attention_mask,
                labels=generated_labels,
            )
        old_logits = None
        if self.kl_coef > 0:
            with torch.no_grad():
                old_outputs = self._non_module_store["old_model"](
                    inputs_embeds=generated_inputs,
                    attention_mask=attention_mask,
                    labels=generated_labels,
                )
            old_logits = old_outputs.logits

        # 阶段 4：GRPO loss
        policy_loss, kl_loss = self._compute_grpo_loss(
            outputs.logits, ref_outputs.logits, old_logits,
            generated_labels,
            targets_metadata, advantages_per_frame, batch_size,
        )
        total_loss = policy_loss + self.kl_coef * kl_loss

        mean_adv = sum(sum(a) for a in advantages_per_frame) / max(
            sum(len(a) for a in advantages_per_frame), 1
        )

        group_mean_rewards, group_var_rewards = [], []
        for b in range(batch_size):
            group_rewards = []
            for g in range(self.G):
                idx = b * self.G + g
                nr = raw_rewards[idx]
                group_rewards.append(sum(nr) / len(nr) if nr else 0.0)
            g_mean = sum(group_rewards) / self.G
            g_var = sum((r - g_mean) ** 2 for r in group_rewards) / self.G
            group_mean_rewards.append(g_mean)
            group_var_rewards.append(g_var)
            if self.debug:
                logging.warning(
                    f"[GRPO-vLLM] b={batch_idx} s={b} "
                    f"group_mean_reward={g_mean:.4f} group_var={g_var:.4f} "
                    f"per_g={[f'{r:.3f}' for r in group_rewards]}"
                )

        self.log_dict({
            "rl/policy_loss": policy_loss,
            "rl/kl_loss": kl_loss,
            "rl/total_loss": total_loss,
            "rl/mean_advantage": mean_adv,
            "rl/group_mean_reward": sum(group_mean_rewards) / max(len(group_mean_rewards), 1),
            "rl/group_reward_var": sum(group_var_rewards) / max(len(group_var_rewards), 1),
        }, prog_bar=True)
        self.log("total_samples", float(self.total_processed_samples),
                 on_step=True, on_epoch=False, prog_bar=False)
        if self.sampler_type not in ("stateless_sampler", "shuffle_queue_stateless_sampler", "shar_stateless_sampler", "shard_pool_sampler"):
            self.log("epoch_samples", float(self.epoch_processed_samples),
                     on_step=True, on_epoch=False, prog_bar=False)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_features, audio_lengths, targets_metadata, prompt_ids, prompt_lens = batch
        batch_size = batch_features.shape[0]

        speech_embeds, speech_embeds_length = self.get_audio_embeds(
            batch_features, audio_lengths
        )
        # 验证时 G=1，贪心解码（temperature=0）
        vllm_states_1 = self.generate_via_vllm(
            speech_embeds, speech_embeds_length,
            prompt_ids, prompt_lens, targets_metadata,
            G=1, temperature=0.0, top_p=1.0,
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
            gen_chunks = vllm_states_1[b]
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

        # 批量计算 COMET
        if self.use_comet and comet_samples:
            comet_scores = self.rewarder.compute_global_comet_batch(comet_samples)
            global_rews = list(comet_scores)

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
                        self.rewarder.compute_process_reward(
                            gen_chunks, target_chunks, t, self.alpha
                        )
                        for t in range(num_chunks)
                    ) / num_chunks
            else:
                reward = 0.0
            raw_rewards.append(reward)

        val_reward = torch.tensor(
            raw_rewards, dtype=torch.float, device=self.device
        ).mean()
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
