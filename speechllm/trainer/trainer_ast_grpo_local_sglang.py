import gc
import logging
import os
import signal
import subprocess
import time

import requests
import torch

from speechllm.trainer.trainer_ast_grpo import SpeechLLMLightningASTGRPO


class SpeechLLMLightningASTGRPOLocalSGLang(SpeechLLMLightningASTGRPO):
    """GRPO trainer，在同一张 GPU 上同时运行训练和 SGLang 推理。

    核心思路：训练和推理分时复用同一张 GPU 的显存。
      - 推理阶段：将训练相关的参数（old_policy、optimizer states）卸载到 CPU，
        腾出显存给 SGLang 做 batch 推理。
      - 训练阶段：SGLang 释放 KV cache，训练参数回到 GPU。

    每个训练 rank 在本地启动一个独立的 SGLang server 子进程，端口为
    base_port + local_rank，生命周期与训练进程绑定。
    """

    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__(model_config, train_ds, val_ds)
        # SGLang 本地实例配置
        self.sglang_base_port = model_config.train.get("sglang_base_port", 30000)
        self.sglang_tp_size = model_config.train.get("sglang_tp_size", 1)
        self.sglang_mem_fraction = model_config.train.get(
            "sglang_mem_fraction", 0.40
        )
        self.sglang_max_running_requests = model_config.train.get(
            "sglang_max_running_requests", 8
        )
        # 是否自动启动 SGLang（设为 False 则假设用户已手动启动）
        self.sglang_auto_launch = model_config.train.get("sglang_auto_launch", True)
        # SGLang 启动超时（秒）
        self.sglang_launch_timeout = model_config.train.get(
            "sglang_launch_timeout", 300
        )
        # 额外传给 sglang 的参数
        self.sglang_extra_args = model_config.train.get("sglang_extra_args", [])
        # 指定 SGLang 使用的 conda 环境名；为 None 时直接用当前环境的 python
        self.sglang_conda_env = model_config.train.get("sglang_conda_env", None)

        self._sglang_process = None  # subprocess.Popen handle

        # 共享内存代理配置
        self.use_shm_proxy = model_config.train.get("use_shm_proxy", False)
        self.shm_proxy_port_offset = model_config.train.get("shm_proxy_port_offset", 100)
        self._shm_proxy_process = None  # subprocess.Popen handle

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """在 sanity check 之前启动 SGLang，确保 validation_step 可用。"""
        super_fit_start = getattr(super(), "on_fit_start", None)
        if super_fit_start:
            super_fit_start()
        # 每个 rank 用 local_rank 分配端口，避免同一节点端口冲突
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        my_port = self.sglang_base_port + local_rank
        self.sglang_server_url = f"http://127.0.0.1:{my_port}"
        self._sglang_port = my_port
        self._sglang_gpu_id = local_rank

        logging.info(
            f"[GPU {self.global_rank}] Local SGLang config: "
            f"port={my_port}, gpu={local_rank}, "
            f"mem_fraction={self.sglang_mem_fraction}"
        )

        if self.sglang_auto_launch:
            self._launch_sglang_server()

    def on_train_start(self) -> None:
        super().on_train_start()

    def on_train_end(self) -> None:
        self._shutdown_sglang_server()
        super_end = getattr(super(), "on_train_end", None)
        if super_end:
            super_end()

    # ------------------------------------------------------------------
    # SGLang server 管理
    # ------------------------------------------------------------------

    def _launch_sglang_server(self):
        """在本地 GPU 上启动 SGLang server 子进程。"""
        # 使用 sglang_weight_sync_path 作为初始模型路径；
        # 如果还没有同步过权重，先做一次初始保存
        model_path = self.sglang_weight_sync_path
        if not os.path.exists(os.path.join(model_path, "config.json")):
            logging.info(
                f"[GPU {self.global_rank}] Saving initial weights for SGLang "
                f"to {model_path}"
            )
            self._save_llm_weights_for_sglang(model_path)

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(self._sglang_port),
            "--tp-size", str(self.sglang_tp_size),
            "--mem-fraction-static", str(self.sglang_mem_fraction),
            "--max-running-requests", str(self.sglang_max_running_requests),
        ]
        cmd.extend(self.sglang_extra_args)

        # 如果指定了 sglang_conda_env，用 conda run 包装命令
        if self.sglang_conda_env:
            cmd = [
                "conda", "run", "-n", self.sglang_conda_env,
                "--no-capture-output",
            ] + cmd

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self._sglang_gpu_id)
        # 清除 PYTHONPATH，避免训练环境的路径污染 SGLang 的 conda 环境
        if self.sglang_conda_env:
            env.pop("PYTHONPATH", None)

        log_dir = os.path.join(
            getattr(self, "exp_dir", "/tmp"), "sglang_logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"sglang_rank{self.global_rank}.log")

        logging.info(
            f"[GPU {self.global_rank}] Launching SGLang: {' '.join(cmd)}"
        )
        log_fh = open(log_path, "w")
        self._sglang_log_fh = log_fh
        self._sglang_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # 新进程组，方便清理
        )

        # 等待 server 就绪
        self._wait_for_sglang_ready()

    def _wait_for_sglang_ready(self):
        """轮询 SGLang /health 端点直到就绪或超时。"""
        url = f"{self.sglang_server_url}/health"
        deadline = time.time() + self.sglang_launch_timeout
        interval = 5
        while time.time() < deadline:
            # 检查子进程是否已退出
            if self._sglang_process and self._sglang_process.poll() is not None:
                raise RuntimeError(
                    f"[GPU {self.global_rank}] SGLang process exited with "
                    f"code {self._sglang_process.returncode} before becoming ready. "
                    f"Check log for details."
                )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logging.info(
                        f"[GPU {self.global_rank}] SGLang server ready at "
                        f"{self.sglang_server_url}"
                    )
                    return
            except requests.ConnectionError:
                pass
            except Exception as e:
                logging.debug(
                    f"[GPU {self.global_rank}] SGLang health check: {e}"
                )
            time.sleep(interval)

        raise RuntimeError(
            f"[GPU {self.global_rank}] SGLang server did not become ready "
            f"within {self.sglang_launch_timeout}s"
        )

    def _shutdown_sglang_server(self):
        """优雅关闭 SGLang server 子进程。"""
        proc = self._sglang_process
        if proc is None:
            return
        logging.info(
            f"[GPU {self.global_rank}] Shutting down SGLang server (pid={proc.pid})"
        )
        try:
            # 先发 SIGTERM 给整个进程组
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logging.warning(
                f"[GPU {self.global_rank}] SGLang did not exit gracefully, "
                f"sending SIGKILL"
            )
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except Exception as e:
            logging.warning(
                f"[GPU {self.global_rank}] Error shutting down SGLang: {e}"
            )
        finally:
            self._sglang_process = None
            if hasattr(self, "_sglang_log_fh"):
                self._sglang_log_fh.close()

    # ------------------------------------------------------------------
    # 显存管理：训练 ↔ 推理切换
    # ------------------------------------------------------------------

    def _offload_training_states_to_cpu(self):
        """将 ref_model、old_model 和 optimizer states 卸载到 CPU，为 SGLang 推理腾出显存。"""
        ref_model = self._non_module_store.get("ref_model", None)
        if ref_model is not None:
            ref_model.cpu()
        old_model = self._non_module_store.get("old_model", None)
        if old_model is not None:
            old_model.cpu()

        torch.cuda.empty_cache()
        gc.collect()

    def _reload_training_states_to_gpu(self):
        """推理结束后，将 ref_model 和 old_model 搬回 GPU。"""
        ref_model = self._non_module_store.get("ref_model", None)
        if ref_model is not None:
            ref_model.to(self.device)
        old_model = self._non_module_store.get("old_model", None)
        if old_model is not None:
            old_model.to(self.device)

    # ------------------------------------------------------------------
    # 权重同步（覆盖基类）
    # ------------------------------------------------------------------

    def _save_llm_weights_for_sglang(self, save_path: str):
        """将 LLM 权重（含 special token patch）保存到指定路径。
        仅在 rank 0 执行实际写盘操作。
        """
        try:
            import deepspeed

            with deepspeed.zero.GatheredParameters(
                list(self.llm_model.parameters()), modifier_rank=0
            ):
                if self.global_rank == 0:
                    # 写入 special token patch
                    if getattr(self, "finetune_special_tokens", False) or getattr(
                        self, "use_lora", False
                    ):
                        special_tokens_list = list(self.special_tokens_dict.keys())
                        input_embeds_weight = (
                            self.llm_model.get_input_embeddings().weight
                        )
                        output_embeds_weight = (
                            self.llm_model.get_output_embeddings().weight
                        )
                        with torch.no_grad():
                            for i, token in enumerate(special_tokens_list):
                                tid = self.special_token_ids[token]
                                input_embeds_weight[tid].copy_(
                                    self.special_token_input_patch.data[i]
                                )
                                output_embeds_weight[tid].copy_(
                                    self.special_token_output_patch.data[i]
                                )
                        logging.info(
                            f"[GPU 0] Applied special_token patches before saving"
                        )

                    os.makedirs(save_path, exist_ok=True)
                    self.llm_model.save_pretrained(save_path)
                    self.llm_tokenizer.save_pretrained(save_path)
                    logging.info(f"[GPU 0] Saved LLM weights to {save_path}")
        except Exception as e:
            logging.warning(
                f"[GPU {self.global_rank}] Save LLM weights error: {e}"
            )

    def _sync_sglang_weights(self):
        """将 LLM 权重保存到磁盘，然后通知本地 SGLang 实例重新加载。

        与 multi_sglang 版本的区别：
        - rank 0 写盘后，所有 rank 各自通知自己本地的 SGLang 实例加载。
        - 使用 barrier 确保 rank 0 写盘完成后其他 rank 才发起加载请求。
        """
        try:
            import deepspeed

            # 所有 rank 先让本地 server flush cache
            try:
                requests.post(
                    f"{self.sglang_server_url}/flush_cache", timeout=30
                )
            except Exception:
                pass

            # rank 0 聚合权重并写盘
            with deepspeed.zero.GatheredParameters(
                list(self.llm_model.parameters()), modifier_rank=0
            ):
                if self.global_rank == 0:
                    if getattr(self, "finetune_special_tokens", False) or getattr(
                        self, "use_lora", False
                    ):
                        special_tokens_list = list(self.special_tokens_dict.keys())
                        input_embeds_weight = (
                            self.llm_model.get_input_embeddings().weight
                        )
                        output_embeds_weight = (
                            self.llm_model.get_output_embeddings().weight
                        )
                        with torch.no_grad():
                            for i, token in enumerate(special_tokens_list):
                                tid = self.special_token_ids[token]
                                input_embeds_weight[tid].copy_(
                                    self.special_token_input_patch.data[i]
                                )
                                output_embeds_weight[tid].copy_(
                                    self.special_token_output_patch.data[i]
                                )
                        logging.info(
                            f"[GPU 0] Applied special_token patches before saving"
                        )

                    self.llm_model.save_pretrained(self.sglang_weight_sync_path)
                    self.llm_tokenizer.save_pretrained(self.sglang_weight_sync_path)
                    logging.info(
                        f"[GPU 0] Saved weights to {self.sglang_weight_sync_path}"
                    )

            # barrier 确保 rank 0 写盘完成
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # 每个 rank 通知自己本地的 SGLang 实例加载新权重
            try:
                resp = requests.post(
                    f"{self.sglang_server_url}/update_weights_from_disk",
                    json={"model_path": self.sglang_weight_sync_path},
                    timeout=300,
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("success"):
                    logging.info(
                        f"[GPU {self.global_rank}] Local SGLang weights synced"
                    )
                else:
                    logging.warning(
                        f"[GPU {self.global_rank}] Local SGLang sync failed: "
                        f"{result.get('message')}"
                    )
            except Exception as e:
                logging.warning(
                    f"[GPU {self.global_rank}] Local SGLang sync error: {e}"
                )

        except Exception as e:
            logging.warning(
                f"[GPU {self.global_rank}] SGLang weight sync error "
                f"(non-fatal): {e}"
            )

    # ------------------------------------------------------------------
    # 覆盖 training_step：在推理前后做显存切换
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """与基类逻辑一致，但在 SGLang 推理前后增加显存管理。"""
        batch_features, audio_lengths, targets_metadata, prompt_ids, prompt_lens = batch
        batch_size = batch_features.shape[0]

        self.total_processed_samples += batch_size
        self.epoch_processed_samples += batch_size

        # π_ref (参考模型) 同步 — 用于 PPO ratio，定期同步
        if (
            self.ref_model_sync_interval > 0 and self._steps_since_ref_sync >= self.ref_model_sync_interval
            or "ref_model" not in self._non_module_store
        ):
            self._sync_ref_model()
            self._steps_since_ref_sync = 0
        self._steps_since_ref_sync += 1

        # π_old (旧模型) 同步 — 用于 KL 散度，old_policy_sync_interval=0 时仅初始化不更新
        if (
            self.old_policy_sync_interval > 0 and self._steps_since_old_sync >= self.old_policy_sync_interval
            or "old_model" not in self._non_module_store
        ):
            self._sync_old_policy()
            self._steps_since_old_sync = 0
        self._steps_since_old_sync += 1

        need_sglang_sync = (
            self._steps_since_sglang_sync >= self.sglang_sync_interval
        )
        if need_sglang_sync:
            self._steps_since_sglang_sync = 0
        self._steps_since_sglang_sync += 1

        # ---- 阶段 1: SGLang 推理（卸载训练状态腾显存）----
        speech_embeds, speech_embeds_length = self.get_audio_embeds(
            batch_features, audio_lengths
        )

        self._offload_training_states_to_cpu()

        with torch.no_grad():
            sglang_states_G = self.generate_via_sglang(
                speech_embeds,
                speech_embeds_length,
                prompt_ids,
                prompt_lens,
                targets_metadata,
                G=self.G,
                batch_idx=batch_idx,
            )

        self._reload_training_states_to_gpu()

        if need_sglang_sync:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self._sync_sglang_weights()

        with torch.no_grad():
            generated_inputs, generated_labels, attention_mask = (
                self.prepare_rl_forward_inputs(
                    speech_embeds, speech_embeds_length,
                    sglang_states_G, targets_metadata,
                    prompt_ids, prompt_lens,
                )
            )

        # ---- 阶段 2: 计算奖励与优势 ----
        normalized_rewards, advantages_per_frame, raw_rewards = (
            self.compute_process_bleu_rewards(
                sglang_states_G, targets_metadata, batch_size
            )
        )

        # ---- 阶段 3: Actor + Ref Model (ratio) + Old Model (KL) 前向 ----
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

        # ---- 阶段 4: GRPO Loss ----
        policy_loss, kl_loss = self._compute_grpo_loss(
            outputs.logits, ref_outputs.logits, old_logits,
            generated_labels,
            targets_metadata, advantages_per_frame, batch_size,
        )
        total_loss = policy_loss + self.kl_coef * kl_loss

        mean_adv = sum(sum(a) for a in advantages_per_frame) / max(
            sum(len(a) for a in advantages_per_frame), 1
        )

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
