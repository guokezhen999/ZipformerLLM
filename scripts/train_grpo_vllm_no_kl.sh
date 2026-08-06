#!/bin/bash

cd /pfs/asr/guokezhen/ZipformerLLM
ROOT_DIR=$(pwd)

source /pfs/asr/miniconda3/etc/profile.d/conda.sh
conda activate zipformer_vllm

# vLLM 推理和训练统一使用 zipformer_vllm 环境
VLLM_PYTHON=/pfs/asr/miniconda3/envs/zipformer_vllm/bin/python
TRAIN_PYTHON=/pfs/asr/miniconda3/envs/zipformer_vllm/bin/python
TRAIN_TORCHRUN=/pfs/asr/miniconda3/envs/zipformer_vllm/bin/torchrun

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export RAY_IGNORE_VERSION_MISMATCH=1

ulimit -n 65536

mkdir -p logs
DATETIME=$(date +%Y%m%d_%H_%M)
MAIN_LOG="logs/main_${DATETIME}.log"
ACTOR_LOG="logs/vllm_actors_${DATETIME}.log"
TRAIN_LOG="logs/train_grpo_${DATETIME}.log"
exec > >(tee -a ${MAIN_LOG}) 2>&1
echo "Logs: main=${MAIN_LOG}, actor=${ACTOR_LOG}, train=${TRAIN_LOG}"

# ============================================================
# 配置
# ============================================================
CONFIG="conf/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000.json"
SFT_CKPT="pretrained_models/stage2/epoch-epoch=14-step-step=194787/epoch-epoch=14-step-step=194787.pt"
VLLM_MODEL_PATH="pretrained_models/stage2/epoch-epoch=14-step-step=194787/llm_epoch-epoch=14-step-step=194787"
VLLM_WEIGHT_SYNC_PATH="/dev/shm/vllm_sync_weights"

# vLLM 推理 GPU 和训练 GPU（4卡，每张卡同时负责推理和训练；COMET 仅在 rank3）
VLLM_GPUS="0,1,2,3"
TRAIN_GPUS="0,1,2,3"

ACTOR_NAME_PREFIX="vllm_actor"
VLLM_MEM_FRACTION=0.1

IFS=',' read -ra VLLM_GPU_ARRAY  <<< "${VLLM_GPUS}"
IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_VLLM=${#VLLM_GPU_ARRAY[@]}
NUM_TRAIN=${#TRAIN_GPU_ARRAY[@]}

export PYTHONPATH=${ROOT_DIR}:$PYTHONPATH

# ============================================================
# 第 1 步：启动 Ray head（使用 4 张 GPU）
# ============================================================
echo "Starting Ray head..."
RAY_PORT=6399
RAY_DASHBOARD_PORT=8266

ray stop --force 2>/dev/null; sleep 2

ray start --head \
    --num-cpus=12 \
    --num-gpus=${NUM_VLLM} \
    --port=${RAY_PORT} \
    --dashboard-port=${RAY_DASHBOARD_PORT} \
    --disable-usage-stats
echo "Ray head started."

# ============================================================
# 第 2 步：启动 vLLM Ray actor（后台运行）
# ============================================================
echo "Launching ${NUM_VLLM} vLLM actors on GPU [${VLLM_GPUS}]..."

${VLLM_PYTHON} speechllm/train/start_vllm_ray_actors.py \
    --model-path   ${VLLM_MODEL_PATH} \
    --gpu-ids      ${VLLM_GPUS} \
    --ray-address  auto \
    --tp-size      1 \
    --max-model-len 4096 \
    --gpu-memory-util ${VLLM_MEM_FRACTION} \
    --actor-name-prefix ${ACTOR_NAME_PREFIX} \
    --dtype bfloat16 \
    --enforce-eager \
    > ${ACTOR_LOG} 2>&1 &
ACTOR_LAUNCHER_PID=$!
echo "  Actor launcher PID: ${ACTOR_LAUNCHER_PID}, log: ${ACTOR_LOG}"

# ============================================================
# 等待所有 vLLM actor 就绪（轮询 Ray actor health，最多 300 秒）
# ============================================================
echo "Waiting for all ${NUM_VLLM} vLLM actors to be ready..."
${VLLM_PYTHON} - <<PYEOF
import ray, time, sys
ray.init(address="auto", ignore_reinit_error=True, namespace="speechllm_vllm")
prefix = "${ACTOR_NAME_PREFIX}"
num    = ${NUM_VLLM}
for i in range(num):
    name = f"{prefix}_{i}"
    deadline = time.time() + 300
    ready = False
    while time.time() < deadline:
        try:
            actor = ray.get_actor(name)
            ok = ray.get(actor.health.remote(), timeout=10)
            if ok:
                print(f"  actor '{name}' ready")
                ready = True
                break
        except Exception as e:
            pass
        time.sleep(5)
    if not ready:
        print(f"ERROR: actor '{name}' not ready within 300s", file=sys.stderr)
        sys.exit(1)
print(f"All {num} vLLM actors ready.")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: vLLM actor startup failed. Check ${ACTOR_LOG}"
    kill ${ACTOR_LAUNCHER_PID} 2>/dev/null
    ray stop
    exit 1
fi

# ============================================================
# 第 3 步：启动 GRPO 训练（仅使用训练 GPU）
# ============================================================
echo "Starting GRPO training on ${NUM_TRAIN} GPUs (${TRAIN_GPUS}) with ${NUM_VLLM} vLLM actors..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} ${TRAIN_TORCHRUN} \
    --nproc_per_node=${NUM_TRAIN} \
    --master_port=29500 \
    speechllm/train/train_grpo_vllm.py \
    --config   ${CONFIG} \
    --grpo_mode multi_vllm \
    --pretrained_model_path ${SFT_CKPT} \
    --debug \
    2>&1 | tee ${TRAIN_LOG}

TRAIN_EXIT=${PIPESTATUS[0]}

# ============================================================
# 训练结束后清理
# ============================================================
echo "Training finished (exit=${TRAIN_EXIT}). Stopping vLLM actors and Ray..."
kill ${ACTOR_LAUNCHER_PID} 2>/dev/null
wait ${ACTOR_LAUNCHER_PID} 2>/dev/null
ray stop
echo "Done."
exit ${TRAIN_EXIT}
