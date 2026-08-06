"""
独立定义 VLLMEmbedActor，不依赖 speechllm.trainer 包，
供 start_vllm_ray_actors.py 在 speechllm_vllm 环境中安全导入。
"""

import logging
import os


def _get_vllm_actor_cls():
    import ray
    from vllm import LLM, SamplingParams

    @ray.remote(num_gpus=1)
    class VLLMEmbedActor:
        def __init__(self, model_path: str, gpu_id: int, tp_size: int = 1,
                     max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                     extra_kwargs: dict = None):
            import torch  # noqa: F401 — ensure torch is loaded in actor process
            import logging
            # Prefer the GPU Ray already bound via CUDA_VISIBLE_DEVICES.
            # Only set an explicit id when the env is unset; overwriting Ray's
            # remapping with a physical id that is not visible (e.g. GPU 1 on a
            # 1-GPU machine) makes CUDA empty and vLLM fails during model inspect.
            if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            else:
                # Keep Ray assignment; log physical request for debugging
                logging.info(
                    f"[VLLMActor] Ray CUDA_VISIBLE_DEVICES="
                    f"{os.environ['CUDA_VISIBLE_DEVICES']} (requested gpu_id={gpu_id})"
                )

            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                raise RuntimeError(
                    f"[VLLMActor] No CUDA device visible "
                    f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                    f"requested gpu_id={gpu_id}). Check VLLM_GPUS / Ray --num-gpus."
                )

            # ── 抑制 vLLM tqdm 进度条输出（不能 disable，vLLM 用 tqdm 计时间）──
            try:
                import tqdm.std
                _devnull = open(os.devnull, "w")
                _orig_init = tqdm.std.tqdm.__init__

                def _patched_init(self, *a, **kw):
                    kw.setdefault("file", _devnull)
                    _orig_init(self, *a, **kw)

                tqdm.std.tqdm.__init__ = _patched_init
            except Exception:
                pass

            for _name in ("vllm", "vllm.engine", "vllm.v1", "vllm.core",
                          "vllm.entrypoints", "vllm.model_executor"):
                logging.getLogger(_name).setLevel(logging.WARNING)

            extra_kwargs = extra_kwargs or {}
            extra_kwargs.setdefault("disable_log_stats", True)
            # Prefix KV reuse for EmbedsPrompt: vLLM hashes prompt_embeds per
            # block (sha256). Consecutive streaming segments that share a
            # common embed prefix will hit cached KV blocks when sticky-routed
            # to the same actor.
            self._llm = LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                enable_prompt_embeds=True,
                enable_prefix_caching=True,
                enable_chunked_prefill=False,
                async_scheduling=False,
                **extra_kwargs,
            )
            logging.info(
                f"[VLLMActor gpu={gpu_id}] ready model={model_path} "
                f"prefix_caching=True"
            )

        @staticmethod
        def _embeds_tensor(input_embeds: list):
            import torch
            # Keep dtype stable across requests so prefix-block hashes match.
            return torch.tensor(input_embeds, dtype=torch.bfloat16)

        @staticmethod
        def _pack_output(req_out) -> dict:
            text = req_out.outputs[0].text if req_out.outputs else ""
            n_cached = getattr(req_out, "num_cached_tokens", None) or 0
            prompt_len = 0
            if getattr(req_out, "prompt_token_ids", None) is not None:
                prompt_len = len(req_out.prompt_token_ids)
            return {
                "text": text,
                "num_cached_tokens": int(n_cached),
                "prompt_len": int(prompt_len),
            }

        def generate(self, input_embeds: list, sampling_params_dict: dict) -> str:
            from vllm import SamplingParams as SP
            from vllm.inputs.llm import EmbedsPrompt

            embeds = self._embeds_tensor(input_embeds)
            sp = SP(**sampling_params_dict)
            prompt = EmbedsPrompt(prompt_embeds=embeds)
            outputs = self._llm.generate(prompt, sp)
            # Keep str return for GRPO trainer compatibility.
            return outputs[0].outputs[0].text

        def generate_batch(self, batch_embeds: list, sampling_params_dict: dict) -> list:
            """Return list of {text, num_cached_tokens, prompt_len}."""
            from vllm import SamplingParams as SP
            from vllm.inputs.llm import EmbedsPrompt

            sp = SP(**sampling_params_dict)
            prompts = [
                EmbedsPrompt(prompt_embeds=self._embeds_tensor(emb_list))
                for emb_list in batch_embeds
            ]
            outputs = self._llm.generate(prompts, sp)
            return [self._pack_output(o) for o in outputs]

        def reload_weights(self, model_path: str) -> bool:
            try:
                # vLLM 0.19+ V1：通过 collective_rpc 原地重载 HF checkpoint 目录
                if hasattr(self._llm, "collective_rpc"):
                    self._llm.collective_rpc(
                        "reload_weights",
                        kwargs={"weights_path": model_path},
                    )
                    logging.warning(
                        f"[VLLMActor] reload_weights OK via collective_rpc "
                        f"from {model_path}"
                    )
                    return True

                # 旧版 v0 引擎回退路径
                engine = self._llm.llm_engine
                engine.model_executor.driver_worker.model_runner.model.load_weights(
                    model_path
                )
                logging.warning(f"[VLLMActor] 已从 {model_path} 重新加载权重 (legacy)")
                return True
            except Exception as e:
                logging.warning(
                    f"[VLLMActor] reload_weights 失败: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                return False

        def health(self) -> bool:
            return True

    return VLLMEmbedActor
