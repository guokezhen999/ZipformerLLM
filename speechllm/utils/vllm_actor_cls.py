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
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
            self._llm = LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                enable_prompt_embeds=True,
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                async_scheduling=False,
                **extra_kwargs,
            )
            logging.info(f"[VLLMActor gpu={gpu_id}] 就绪，model={model_path}")
        def generate(self, input_embeds: list, sampling_params_dict: dict) -> str:
            import torch
            from vllm import SamplingParams as SP
            from vllm.inputs.llm import EmbedsPrompt

            embeds = torch.tensor(input_embeds, dtype=torch.bfloat16)
            sp = SP(**sampling_params_dict)
            prompt = EmbedsPrompt(prompt_embeds=embeds)
            outputs = self._llm.generate(prompt, sp)
            return outputs[0].outputs[0].text

        def generate_batch(self, batch_embeds: list, sampling_params_dict: dict) -> list:
            import torch
            from vllm import SamplingParams as SP
            from vllm.inputs.llm import EmbedsPrompt

            sp = SP(**sampling_params_dict)
            prompts = []
            for emb_list in batch_embeds:
                embeds = torch.tensor(emb_list, dtype=torch.float16)
                prompts.append(EmbedsPrompt(prompt_embeds=embeds))
            outputs = self._llm.generate(prompts, sp)
            return [o.outputs[0].text for o in outputs]

        def reload_weights(self, model_path: str) -> bool:
            try:
                # vLLM 0.19.x API: LLM.llm_engine → LLMEngine.model_executor
                engine = self._llm.llm_engine
                engine.model_executor.driver_worker.model_runner.model.load_weights(model_path)
                logging.info(f"[VLLMActor] 已从 {model_path} 重新加载权重")
                return True
            except Exception as e:
                logging.debug(f"[VLLMActor] reload_weights 失败: {e}")
                return False

        def health(self) -> bool:
            return True

    return VLLMEmbedActor
