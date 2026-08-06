#!/bin/bash
# Launch multi-user streaming ASR+AST app with shared vLLM decode.
# Environment: zipformer_vllm
#
# Architecture:
#   GPU(s) for vLLM Ray actors  — LLM decode (batched across users / ASR+AST)
#   SPEECHLLM_DEVICE            — Zipformer encoder + connector (+ optional CPU LLM embedder)
#
# Model paths (env only; deploy JSON has architecture, no weight paths):
#   SPEECHLLM_CONFIG      — conf/deploy_model.json
#   SPEECHLLM_CHECKPOINT  — encoder + connector .pt
#   SPEECHLLM_LLM_PATH    — HF LLM export (also default for VLLM_MODEL_PATH)
#
# Export LLM first (once) if missing:
#   bash scripts/app/export_llm.sh
#
# Usage:
#   bash scripts/app/run_app_vllm.sh

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

source /pfs/asr/miniconda3/etc/profile.d/conda.sh
conda activate zipformer_vllm

VLLM_PYTHON=/pfs/asr/miniconda3/envs/zipformer_vllm/bin/python
UVICORN=/pfs/asr/miniconda3/envs/zipformer_vllm/bin/uvicorn

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_IGNORE_VERSION_MISMATCH=1

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p logs
DATETIME=$(date +%Y%m%d_%H_%M)
MAIN_LOG="logs/app_vllm_${DATETIME}.log"
ACTOR_LOG="logs/app_vllm_actors_${DATETIME}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1
echo "Logs: main=${MAIN_LOG}, actor=${ACTOR_LOG}"

# ============================================================
# Model / server config
# ============================================================
export SPEECHLLM_CONFIG="${SPEECHLLM_CONFIG:-conf/deploy_model.json}"
export SPEECHLLM_CHECKPOINT="${SPEECHLLM_CHECKPOINT:-exp/grpo_vllm_kl_0.05_lr_1e-6_comet_step_2000/checkpoints/best-step-1850-weights.pt}"
export SPEECHLLM_LLM_PATH="${SPEECHLLM_LLM_PATH:-pretrained_models/grpo_vllm_kl_0.05_lr_1e-6_comet_step_2000_llm}"
export SPEECHLLM_DEVICE="${SPEECHLLM_DEVICE:-cuda:0}"
export SPEECHLLM_OFFLOAD_LLM="${SPEECHLLM_OFFLOAD_LLM:-1}"

export SPEECHLLM_AST_LANG_OPTIONS="${SPEECHLLM_AST_LANG_OPTIONS:-Chinese,English}"
export SPEECHLLM_ASR_LANG_OPTIONS="${SPEECHLLM_ASR_LANG_OPTIONS:-auto,Chinese,English}"
export SPEECHLLM_REPETITION_PENALTY="${SPEECHLLM_REPETITION_PENALTY:-1.0}"
# no windowed penalty; 1.0 disables repetition penalty for both local and vLLM
export SPEECHLLM_MAX_NEW_TOKENS="${SPEECHLLM_MAX_NEW_TOKENS:-200}"
export SPEECHLLM_MAX_CONNECTIONS="${SPEECHLLM_MAX_CONNECTIONS:-16}"
# KV / history: trigger at max_segments, then keep last keep_segments (not full clear)
export SPEECHLLM_KEEP_SEGMENTS="${SPEECHLLM_KEEP_SEGMENTS:-16}"
# export SPEECHLLM_MAX_SEGMENTS=64   # optional override; default 64//num_chunks

export SPEECHLLM_USE_VAD="${SPEECHLLM_USE_VAD:-1}"
export SPEECHLLM_VAD_THRESHOLD="${SPEECHLLM_VAD_THRESHOLD:-0.5}"
export SPEECHLLM_VAD_MIN_SILENCE_DURATION_MS="${SPEECHLLM_VAD_MIN_SILENCE_DURATION_MS:-250}"
export SPEECHLLM_VAD_SPEECH_PAD_MS="${SPEECHLLM_VAD_SPEECH_PAD_MS:-150}"

# vLLM actors (HF LLM export; default = SPEECHLLM_LLM_PATH)
# NOTE: gpu ids must exist on this machine. This host often has only GPU 0 —
# setting a missing id (e.g. 1) makes CUDA invisible and vLLM fails with a
# misleading "Qwen3ForCausalLM failed to be inspected" error.
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-${SPEECHLLM_LLM_PATH}}"
VLLM_GPUS="${VLLM_GPUS:-0}"          # comma-separated physical GPU ids for vLLM
VLLM_MEM_FRACTION="${VLLM_MEM_FRACTION:-0.1}"
ACTOR_NAME_PREFIX="${ACTOR_NAME_PREFIX:-vllm_actor}"
RAY_PORT="${RAY_PORT:-6399}"
APP_PORT="${APP_PORT:-8002}"
APP_HOST="${APP_HOST:-0.0.0.0}"

IFS=',' read -ra VLLM_GPU_ARRAY <<< "${VLLM_GPUS}"
NUM_VLLM=${#VLLM_GPU_ARRAY[@]}

# Preflight: reject GPU ids that are not visible to this process
NUM_SYS_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ -z "${NUM_SYS_GPUS}" || "${NUM_SYS_GPUS}" -eq 0 ]]; then
    echo "ERROR: no NVIDIA GPU detected by nvidia-smi"
    exit 1
fi
for g in "${VLLM_GPU_ARRAY[@]}"; do
    if (( g < 0 || g >= NUM_SYS_GPUS )); then
        echo "ERROR: VLLM_GPUS id ${g} out of range (machine has ${NUM_SYS_GPUS} GPU(s): 0..$((NUM_SYS_GPUS-1)))"
        echo "  Fix: VLLM_GPUS=0 bash scripts/app/run_app_vllm.sh"
        exit 1
    fi
done
# Ray must see enough GPU indices for the physical ids used by actors
MAX_VLLM_GPU=0
for g in "${VLLM_GPU_ARRAY[@]}"; do
    if (( g > MAX_VLLM_GPU )); then MAX_VLLM_GPU=$g; fi
done
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((MAX_VLLM_GPU + 1))}"

export SPEECHLLM_VLLM_NUM_ACTORS="${NUM_VLLM}"
export SPEECHLLM_VLLM_ACTOR_PREFIX="${ACTOR_NAME_PREFIX}"
export SPEECHLLM_RAY_ADDRESS="${SPEECHLLM_RAY_ADDRESS:-auto}"
export SPEECHLLM_VLLM_BATCH_TIMEOUT_MS="${SPEECHLLM_VLLM_BATCH_TIMEOUT_MS:-20}"
export SPEECHLLM_VLLM_MAX_BATCH="${SPEECHLLM_VLLM_MAX_BATCH:-32}"
export SPEECHLLM_VLLM_TEMPERATURE="${SPEECHLLM_VLLM_TEMPERATURE:-0.0}"

if [[ ! -d "${SPEECHLLM_LLM_PATH}" ]]; then
    echo "ERROR: SPEECHLLM_LLM_PATH not found: ${SPEECHLLM_LLM_PATH}"
    echo "  Export first: bash scripts/app/export_llm.sh"
    exit 1
fi

echo "Encoder device : ${SPEECHLLM_DEVICE} (offload_llm=${SPEECHLLM_OFFLOAD_LLM})"
echo "vLLM GPUs      : [${VLLM_GPUS}]  actors=${NUM_VLLM}  mem=${VLLM_MEM_FRACTION}  ray_gpus=${RAY_NUM_GPUS}"
echo "Config         : ${SPEECHLLM_CONFIG}"
echo "Checkpoint     : ${SPEECHLLM_CHECKPOINT}"
echo "LLM (HF)       : ${SPEECHLLM_LLM_PATH}"
echo "vLLM model     : ${VLLM_MODEL_PATH}"
echo "App            : http://${APP_HOST}:${APP_PORT}"
echo "Max clients    : ${SPEECHLLM_MAX_CONNECTIONS}"

cleanup() {
    echo "Cleaning up..."
    if [[ -n "${UVICORN_PID:-}" ]]; then
        kill "${UVICORN_PID}" 2>/dev/null || true
        wait "${UVICORN_PID}" 2>/dev/null || true
    fi
    if [[ -n "${ACTOR_LAUNCHER_PID:-}" ]]; then
        kill "${ACTOR_LAUNCHER_PID}" 2>/dev/null || true
        wait "${ACTOR_LAUNCHER_PID}" 2>/dev/null || true
    fi
    ray stop --force 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ============================================================
# 1) Ray head
# ============================================================
echo "Starting Ray head..."
ray stop --force 2>/dev/null || true
sleep 2

ray start --head \
    --num-cpus=8 \
    --num-gpus="${RAY_NUM_GPUS}" \
    --port="${RAY_PORT}" \
    --dashboard-port=8266 \
    --disable-usage-stats
echo "Ray head started."

# ============================================================
# 2) vLLM Ray actors
# ============================================================
echo "Launching ${NUM_VLLM} vLLM actors on GPU [${VLLM_GPUS}]..."
${VLLM_PYTHON} speechllm/train/start_vllm_ray_actors.py \
    --model-path   "${VLLM_MODEL_PATH}" \
    --gpu-ids      "${VLLM_GPUS}" \
    --ray-address  auto \
    --tp-size      1 \
    --max-model-len 4096 \
    --gpu-memory-util "${VLLM_MEM_FRACTION}" \
    --actor-name-prefix "${ACTOR_NAME_PREFIX}" \
    --dtype bfloat16 \
    --enforce-eager \
    > "${ACTOR_LOG}" 2>&1 &
ACTOR_LAUNCHER_PID=$!
echo "  Actor launcher PID: ${ACTOR_LAUNCHER_PID}, log: ${ACTOR_LOG}"

echo "Waiting for all ${NUM_VLLM} vLLM actors..."
${VLLM_PYTHON} - <<PYEOF
import ray, time, sys
ray.init(address="auto", ignore_reinit_error=True, namespace="speechllm_vllm")
prefix = "${ACTOR_NAME_PREFIX}"
num = ${NUM_VLLM}
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
        except Exception:
            pass
        time.sleep(5)
    if not ready:
        print(f"ERROR: actor '{name}' not ready within 300s", file=sys.stderr)
        sys.exit(1)
print(f"All {num} vLLM actors ready.")
PYEOF

# ============================================================
# 3) FastAPI / uvicorn
# ============================================================
echo "Starting uvicorn on ${APP_HOST}:${APP_PORT} ..."
${UVICORN} speechllm.app.stream_asr_ast_vllm.server:app \
    --host "${APP_HOST}" \
    --port "${APP_PORT}" \
    --workers 1 \
    &
UVICORN_PID=$!
echo "  uvicorn PID: ${UVICORN_PID}"

wait "${UVICORN_PID}"
