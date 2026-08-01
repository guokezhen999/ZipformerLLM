#!/bin/bash
#SBATCH --job-name=ast_grpo_multi
#SBATCH --output=logs/slurm/grpo_%j.out
#SBATCH --error=logs/slurm/grpo_%j.err
#SBATCH --nodelist=node20
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=480000:00:00

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

ROOT_DIR="/nfs/guokezhen/ZipformerLLM"
cd "${ROOT_DIR}"

source /nfs/asr/guokezhen/miniconda/etc/profile.d/conda.sh
conda activate speechllm

# 配置 CUDA 环境，以防止 FlashInfer 编译时使用老版本系统 NVCC (V11.5)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export FLASHINFER_NVCC=/usr/local/cuda-12.4/bin/nvcc

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 65536
mkdir -p logs/sglang

# ============================================================
# 配置
# ============================================================
CONFIG="${ROOT_DIR}/conf/grpo_sglang_kl_0.05_lr_1e-6_comet_step_1000.json"

# 必填：单个 Lightning DeepSpeed checkpoint 目录（例如 step-step=100000.ckpt）
# 可通过环境变量覆盖：DS_CKPT=/path/to/xxx.ckpt sbatch scripts/train_grpo_sglang.sh
DS_CKPT="${DS_CKPT:-exp/sft_stage2_asr_ast_step_200k/checkpoints/epoch-epoch=14-step-step=194787.ckpt}"
# 转换产物目录；默认写到 DS_CKPT 同级的 converted/<ckpt_name>/
BASE_LLM_MODEL="${ROOT_DIR}/pretrained_models/Qwen3-0.6B"
OVERWRITE_CONVERT="${OVERWRITE_CONVERT:-0}"

# SGLang 推理服务配置
# SGLANG_GPUS: 用于推理的 GPU 列表（逗号分隔）
# TRAIN_GPUS: 用于训练的 GPU 列表（逗号分隔）
# 每个推理 GPU 启动一个 TP=1 的 SGLang 服务
# 训练 rank 通过 rank % num_sglang 分配到对应的推理服务
SGLANG_GPUS="0,1"
TRAIN_GPUS="2,3,4,5,6,7"
BASE_PORT=30000
MEM_FRACTION=0.60

# 解析 GPU 列表
IFS=',' read -ra SGLANG_GPU_ARRAY <<< "${SGLANG_GPUS}"
IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_SGLANG=${#SGLANG_GPU_ARRAY[@]}
NUM_TRAIN=${#TRAIN_GPU_ARRAY[@]}

# ============================================================
# 第 0 步：从单个 DeepSpeed ckpt 转换整模 .pt + SGLang HF LLM
# ============================================================
if [[ -z "${DS_CKPT}" ]]; then
    echo "ERROR: DS_CKPT is required."
    exit 1
fi

# 相对路径按 ROOT_DIR 解析
if [[ "${DS_CKPT}" != /* ]]; then
    DS_CKPT="${ROOT_DIR}/${DS_CKPT}"
fi

if [[ ! -d "${DS_CKPT}" ]]; then
    echo "ERROR: DS_CKPT is not a directory: ${DS_CKPT}"
    exit 1
fi

CKPT_NAME="$(basename "${DS_CKPT}")"
CKPT_STEM="${CKPT_NAME%.ckpt}"
CONVERT_DIR="${CONVERT_DIR:-$(dirname "${DS_CKPT}")/converted/${CKPT_STEM}}"
SFT_CKPT="${CONVERT_DIR}/${CKPT_STEM}.pt"
SGLANG_MODEL_PATH="${CONVERT_DIR}/llm_${CKPT_STEM}"

mkdir -p "${CONVERT_DIR}"

echo "DS_CKPT=${DS_CKPT}"
echo "CONVERT_DIR=${CONVERT_DIR}"
echo "SFT_CKPT=${SFT_CKPT}"
echo "SGLANG_MODEL_PATH=${SGLANG_MODEL_PATH}"

CONVERT_FLAGS=()
if [[ "${OVERWRITE_CONVERT}" == "1" ]]; then
    CONVERT_FLAGS+=(--overwrite)
fi

if [[ ! -f "${SFT_CKPT}" || "${OVERWRITE_CONVERT}" == "1" ]]; then
    echo "Converting DeepSpeed ckpt -> ${SFT_CKPT} ..."
    python speechllm/utils/convert_ds_ckpt.py \
        "${DS_CKPT}" \
        -o "${SFT_CKPT}" \
        "${CONVERT_FLAGS[@]}"
else
    echo "Reuse existing converted ckpt: ${SFT_CKPT}"
fi

if [[ ! -d "${SGLANG_MODEL_PATH}" || "${OVERWRITE_CONVERT}" == "1" ]]; then
    echo "Exporting LLM HF weights -> ${SGLANG_MODEL_PATH} ..."
    rm -rf "${SGLANG_MODEL_PATH}"
    python speechllm/utils/export_sft_llm.py \
        --ckpt "${SFT_CKPT}" \
        --base_model "${BASE_LLM_MODEL}" \
        --output "${SGLANG_MODEL_PATH}"
else
    echo "Reuse existing SGLang model: ${SGLANG_MODEL_PATH}"
fi

# ============================================================
# 第 1 步：启动 SGLang 推理服务（每个推理 GPU 一个，后台）
# ============================================================
SGLANG_PIDS=()

for idx in $(seq 0 $((NUM_SGLANG - 1))); do
    gpu_id=${SGLANG_GPU_ARRAY[$idx]}
    port=$((BASE_PORT + idx))
    log_file="logs/sglang/sglang_${SLURM_JOB_ID}_gpu${gpu_id}.log"

    echo "Starting SGLang instance ${idx} on GPU ${gpu_id}, port ${port}..."
    CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n speechllm_sglang --no-capture-output \
        python -m sglang.launch_server \
        --model-path ${SGLANG_MODEL_PATH} \
        --port ${port} \
        --dtype bfloat16 \
        --tp-size 1 \
        --mem-fraction-static ${MEM_FRACTION} \
        --disable-radix-cache > ${log_file} 2>&1 &

    SGLANG_PIDS+=($!)
    echo "  PID: ${SGLANG_PIDS[-1]}, log: ${log_file}"
done

# ============================================================
# 等待所有 SGLang 服务就绪（最多 600 秒）
# ============================================================
echo "Waiting for all ${NUM_SGLANG} SGLang servers to be ready..."
for idx in $(seq 0 $((NUM_SGLANG - 1))); do
    port=$((BASE_PORT + idx))
    ready=0
    for i in $(seq 1 600); do
        if curl -s http://127.0.0.1:${port}/health > /dev/null 2>&1; then
            echo "  Instance ${idx} (port ${port}) ready after ${i}s."
            ready=1
            break
        fi
        sleep 1
    done
    if [ ${ready} -eq 0 ]; then
        echo "ERROR: SGLang instance ${idx} (port ${port}) failed to start within 600s."
        tail -30 logs/sglang/sglang_${SLURM_JOB_ID}_gpu${SGLANG_GPU_ARRAY[$idx]}.log
        for pid in "${SGLANG_PIDS[@]}"; do
            kill ${pid} 2>/dev/null
        done
        exit 1
    fi
done
echo "All ${NUM_SGLANG} SGLang servers are ready. Each serves ${NUM_TRAIN}/${NUM_SGLANG} training ranks."

# ============================================================
# 第 2 步：启动 GRPO 训练（仅使用训练 GPU）
# ============================================================
# 检查参数或环境变量是否开启 offload
OFFLOAD="${OFFLOAD:-0}"
for arg in "$@"; do
    if [[ "${arg}" == "--offload" ]]; then
        OFFLOAD=1
    fi
done

TRAIN_FLAGS=()
if [[ "${OFFLOAD}" == "1" ]]; then
    echo "DeepSpeed optimizer offloading is ENABLED."
    TRAIN_FLAGS+=(--offload)
fi

echo "Starting GRPO training on ${NUM_TRAIN} GPUs (${TRAIN_GPUS}), with ${NUM_SGLANG} SGLang instances..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} torchrun \
    --nproc_per_node=${NUM_TRAIN} \
    --master_port=29500 \
    speechllm/train/train_grpo_sglang.py \
    --config ${CONFIG} \
    --grpo_mode multi_sglang \
    --pretrained_model_path ${SFT_CKPT} \
    --debug \
    "${TRAIN_FLAGS[@]}"

# ============================================================
# 训练结束后清理所有 SGLang 服务
# ============================================================
echo "Training finished. Stopping all SGLang servers..."
for pid in "${SGLANG_PIDS[@]}"; do
    kill ${pid} 2>/dev/null
    wait ${pid} 2>/dev/null
done
echo "Done."
