"""
启动并保持多个命名 vLLM Ray actor 存活。

用法：
    python -m speechllm.bin.start_vllm_ray_actors \
        --model-path /path/to/llm \
        --gpu-ids 0,1 \
        --ray-address auto

每个 actor 以 Ray detached actor 形式注册，名称为 "{actor_name_prefix}_{i}"。
注意：detached actor 不会随本进程退出自动销毁，退出前必须 ray.kill；
脚本收到 SIGTERM/SIGINT 时会显式销毁全部 actor 以释放显存。
"""

import argparse
import logging
import os
import signal
import sys
import time

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="启动命名 vLLM Ray actor")
    parser.add_argument("--model-path", required=True, help="HF 模型路径")
    parser.add_argument("--gpu-ids", required=True,
                        help="逗号分隔的 GPU 编号列表，如 '0,1'")
    parser.add_argument("--ray-address", default=None,
                        help="Ray 集群地址（默认 None，本地启动）")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="每个 actor 的 tensor parallel size（默认 1）")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="vLLM 最大序列长度（默认 4096）")
    parser.add_argument("--gpu-memory-util", type=float, default=0.85,
                        help="vLLM 显存占用比例（默认 0.85）")
    parser.add_argument("--actor-name-prefix", default="vllm_actor",
                        help="actor 命名前缀，第 i 个 actor 名为 prefix_i（默认 vllm_actor）")
    parser.add_argument("--dtype", default="bfloat16",
                        help="模型精度（默认 bfloat16）")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="禁用 CUDA graph，节省显存（共享 GPU 时推荐）")
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    num_actors = len(gpu_ids)

    import ray
    ray.init(address=args.ray_address, ignore_reinit_error=True, namespace="speechllm_vllm")
    logging.info(f"Ray 已初始化（address={args.ray_address}）")

    # 从独立模块导入，避免触发 speechllm.trainer 包（含 lightning/sklearn 依赖）
    import importlib.util, pathlib
    _spec = importlib.util.spec_from_file_location(
        "vllm_actor_cls",
        pathlib.Path(__file__).parent.parent / "utils" / "vllm_actor_cls.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _get_vllm_actor_cls = _mod._get_vllm_actor_cls
    VLLMEmbedActor = _get_vllm_actor_cls()

    actors = []
    for i, gpu_id in enumerate(gpu_ids):
        name = f"{args.actor_name_prefix}_{i}"
        logging.info(f"正在启动 actor '{name}'（GPU {gpu_id}）...")
        actor = VLLMEmbedActor.options(
            name=name,
            lifetime="detached",  # 需显式 ray.kill；见下方 _kill_actors
            num_gpus=1,
        ).remote(
            model_path=args.model_path,
            gpu_id=gpu_id,
            tp_size=args.tp_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_util,
            extra_kwargs={"dtype": args.dtype, "enforce_eager": args.enforce_eager},
        )
        actors.append((name, actor))

    # 等待所有 actor 就绪（最多 300 秒）
    for name, actor in actors:
        deadline = time.time() + 300
        ready = False
        while time.time() < deadline:
            try:
                ok = ray.get(actor.health.remote(), timeout=10)
                if ok:
                    logging.info(f"  actor '{name}' 就绪")
                    ready = True
                    break
            except Exception as e:
                logging.debug(f"  actor '{name}' 等待中: {e}")
            time.sleep(5)
        if not ready:
            logging.error(f"actor '{name}' 在 300s 内未就绪，退出")
            sys.exit(1)

    logging.info(
        f"所有 {num_actors} 个 vLLM actor 已就绪：" +
        "，".join(f"{name}(GPU {gpu_ids[i]})" for i, (name, _) in enumerate(actors))
    )

    def _kill_actors():
        """detached actor 不会随 driver 退出自动销毁，必须显式 ray.kill 才能释放显存。"""
        for name, actor in actors:
            try:
                ray.kill(actor, no_restart=True)
                logging.info(f"  已销毁 actor '{name}'")
            except Exception as e:
                logging.warning(f"  销毁 actor '{name}' 失败: {e}")

    # 保持进程存活，直到收到 SIGTERM / SIGINT
    stop = [False]

    def _handle_signal(signum, frame):
        logging.info(f"收到信号 {signum}，正在销毁 vLLM actors...")
        stop[0] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not stop[0]:
            time.sleep(2)
    finally:
        _kill_actors()
        logging.info("start_vllm_ray_actors 退出。")


if __name__ == "__main__":
    main()
