import logging
import requests
import torch

from speechllm.trainer.trainer_ast_grpo import SpeechLLMLightningASTGRPO


class SpeechLLMLightningASTGRPOMultiSGLang(SpeechLLMLightningASTGRPO):
    """GRPO trainer，多个训练 rank 可共享同一个 SGLang 推理服务。

    支持 N 个 SGLang 实例服务 M 个训练 rank（M >= N）。
    每个 rank 通过 rank % num_instances 分配到对应的 SGLang 端口。
    例如：2 个 SGLang + 6 个训练 rank → 每 3 个 rank 共享一个 SGLang。
    """

    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__(model_config, train_ds, val_ds)
        # 基础端口，每个 rank 的端口 = base_port + rank
        self.sglang_base_port = model_config.train.get("sglang_base_port", 30000)
        self.sglang_num_instances = model_config.train.get("sglang_num_instances", 8)
        # sglang_server_url 会在 on_train_start 中根据 rank 设置
        # 先保留基类设置的值作为 fallback

    def on_train_start(self) -> None:
        # 在调用 super() 之前先设置好 URL，避免 super() 用错误端口启动 proxy 后再重启
        instance_idx = self.global_rank % self.sglang_num_instances
        my_port = self.sglang_base_port + instance_idx
        self.sglang_server_url = f"http://127.0.0.1:{my_port}"
        logging.info(
            f"[GPU {self.global_rank}] Using SGLang instance {instance_idx} "
            f"at {self.sglang_server_url}"
        )
        super().on_train_start()

    def _sync_sglang_weights(self):
        """将 LLM 权重保存到磁盘，然后通知所有 SGLang 实例重新加载。"""
        try:
            import deepspeed

            # rank 0 先通知所有 server flush cache
            if self.global_rank == 0:
                for i in range(self.sglang_num_instances):
                    port = self.sglang_base_port + i
                    try:
                        requests.post(
                            f"http://127.0.0.1:{port}/flush_cache", timeout=30
                        )
                    except Exception:
                        pass

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

                    self.llm_model.save_pretrained(self.sglang_weight_sync_path)
                    self.llm_tokenizer.save_pretrained(self.sglang_weight_sync_path)

                    # 通知所有 SGLang 实例加载新权重
                    for i in range(self.sglang_num_instances):
                        port = self.sglang_base_port + i
                        url = f"http://127.0.0.1:{port}"
                        try:
                            resp = requests.post(
                                f"{url}/update_weights_from_disk",
                                json={
                                    "model_path": self.sglang_weight_sync_path
                                },
                                timeout=300,
                            )
                            resp.raise_for_status()
                            result = resp.json()
                            if result.get("success"):
                                logging.info(
                                    f"[GPU 0] SGLang instance {i} (port {port}) weights synced"
                                )
                            else:
                                logging.warning(
                                    f"[GPU 0] SGLang instance {i} sync failed: {result.get('message')}"
                                )
                        except Exception as e:
                            logging.warning(
                                f"[GPU 0] SGLang instance {i} sync error: {e}"
                            )
        except Exception as e:
            logging.warning(
                f"[GPU {self.global_rank}] SGLang weight sync error (non-fatal): {e}"
            )
