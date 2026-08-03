"""
多 vLLM 实例 GRPO 训练器。

架构：N 个 vLLM Ray actor 预先由外部脚本（start_vllm_ray_actors.py）在专用 GPU 上
启动，训练侧通过 ray.get_actor(name) 连接到已有 actor，不自行创建也不销毁。

路由规则：training_rank % vllm_num_actors → 选取对应 actor 索引。
例如：2 个 actor + 6 个训练 rank → 每 3 个 rank 共享同一个 actor。

与 trainer_ast_grpo_vllm.py 的区别：
  - on_fit_start  只连接已有 actor，不启动新 actor。
  - on_train_end  不销毁 actor（生命周期由外部脚本管理）。
  - _sync_vllm_weights  由 rank 0 写盘后，rank 0 依次通知所有 actor 重载权重。
"""

import logging
import os
import time

import torch

from speechllm.trainer.trainer_ast_grpo_vllm import SpeechLLMLightningASTGRPOVLLM


class SpeechLLMLightningASTGRPOMultiVLLM(SpeechLLMLightningASTGRPOVLLM):
    """多 vLLM actor 版 GRPO 训练器。

    配置项（位于 ``train`` 下，在基类基础上新增/覆盖）：
        vllm_num_actors       : vLLM actor 总数（默认 2）
        vllm_actor_name_prefix: Ray 命名 actor 的前缀（默认 "vllm_actor"）
                                第 i 个 actor 的名称为 "{prefix}_{i}"
    """

    def __init__(self, model_config=None, train_ds=None, val_ds=None):
        super().__init__(model_config, train_ds, val_ds)
        tcfg = model_config.train
        self.vllm_num_actors = tcfg.get("vllm_num_actors", 2)
        self.vllm_actor_name_prefix = tcfg.get("vllm_actor_name_prefix", "vllm_actor")
        # _vllm_actors: 所有 actor 的句柄列表（仅 rank 0 在 _sync_vllm_weights 中需要完整列表）
        self._vllm_actors_all = []

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """只初始化 Ray 并连接到已有 actor，不启动新 actor。"""
        super_hook = getattr(super(SpeechLLMLightningASTGRPOVLLM, self), "on_fit_start", None)
        if super_hook:
            super_hook()
        self._init_ray()
        self._connect_to_vllm_actors()

    def on_train_start(self) -> None:
        """跳过基类中启动 SGLang proxy 逻辑，但训练前先同步 vLLM 权重。"""
        self._resume_global_step = 0
        if not self.finetune_encoder:
            self.audio_encoder.eval()

        # 仅在 comet_rank（或未指定时所有 rank）预热 COMET
        self._warmup_comet_if_needed()

        # 外部预启动的 vLLM actor 可能加载的不是当前训练 checkpoint
        # 中的 LLM / special token 权重，第一批生成前必须先同步一次。
        self._sync_vllm_weights()
        self._steps_since_vllm_sync = 0

    def on_train_end(self) -> None:
        """不销毁 actor，生命周期由外部 start_vllm_ray_actors.py 管理。"""
        pass

    # ------------------------------------------------------------------
    # 连接到已有命名 actor
    # ------------------------------------------------------------------

    def _connect_to_vllm_actors(self):
        """通过命名获取本 rank 对应的 vLLM actor 句柄。"""
        import ray

        # 当前 rank 分配到哪个 actor
        instance_idx = self.global_rank % self.vllm_num_actors
        actor_name = f"{self.vllm_actor_name_prefix}_{instance_idx}"

        # 最多等待 300 秒
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                self._vllm_actor = ray.get_actor(actor_name, namespace="speechllm_vllm")
                # 健康检查
                ok = ray.get(self._vllm_actor.health.remote(), timeout=10)
                if ok:
                    logging.info(
                        f"[GPU {self.global_rank}] 已连接到 Ray actor "
                        f"'{actor_name}'（instance_idx={instance_idx}）"
                    )
                    break
            except Exception as e:
                logging.debug(
                    f"[GPU {self.global_rank}] 等待 actor '{actor_name}': {e}"
                )
            time.sleep(5)
        else:
            raise RuntimeError(
                f"[GPU {self.global_rank}] 在 300s 内未能连接到 actor '{actor_name}'"
            )

        # rank 0 还需要持有所有 actor 的句柄，用于权重广播
        if self.global_rank == 0:
            self._vllm_actors_all = []
            for i in range(self.vllm_num_actors):
                name = f"{self.vllm_actor_name_prefix}_{i}"
                try:
                    actor = ray.get_actor(name, namespace="speechllm_vllm")
                    self._vllm_actors_all.append(actor)
                    logging.info(f"[GPU 0] 已获取 actor '{name}' 句柄")
                except Exception as e:
                    logging.warning(f"[GPU 0] 获取 actor '{name}' 失败: {e}")

    # ------------------------------------------------------------------
    # 权重同步（覆盖基类：rank 0 通知所有 actor）
    # ------------------------------------------------------------------

    def _sync_vllm_weights(self):
        """保存权重后，由 rank 0 通知所有 vLLM actor 重新加载。"""
        import ray
        try:
            self._save_llm_weights_for_vllm(self.vllm_weight_sync_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # 只有 rank 0 负责通知所有 actor，避免重复写盘或并发冲突
            if self.global_rank == 0:
                for i, actor in enumerate(self._vllm_actors_all):
                    try:
                        ok = ray.get(
                            actor.reload_weights.remote(self.vllm_weight_sync_path),
                            timeout=300,
                        )
                        if ok:
                            logging.info(
                                f"[GPU 0] vLLM actor {i} 权重已从 "
                                f"{self.vllm_weight_sync_path} 重新加载"
                            )
                        else:
                            logging.warning(
                                f"[GPU 0] vLLM actor {i} reload_weights 返回 False"
                            )
                    except Exception as e:
                        logging.warning(f"[GPU 0] vLLM actor {i} 同步出错: {e}")
        except Exception as e:
            logging.warning(
                f"[GPU {self.global_rank}] vLLM 权重同步出错（非致命）: {e}"
            )
