#!/bin/bash
# Stream ASR + AST parallel web demo (English ASR, English→Chinese AST)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="https://hf-mirror.com"

source /nfs/asr/guokezhen/miniconda/etc/profile.d/conda.sh
conda activate speechllm

# Language options: ASR English only; AST target Chinese (EN→ZH)
export SPEECHLLM_AST_LANG_OPTIONS="Chinese"
export SPEECHLLM_ASR_LANG_OPTIONS="English"

export SPEECHLLM_CONFIG="conf/grpo_vllm_kl_0.05_lr_1e-6_comet_step_2000.json"
export SPEECHLLM_CHECKPOINT="exp/grpo_vllm_kl_0.05_lr_1e-6_comet_step_2000/checkpoints/best-step-1850-weights.pt"
export SPEECHLLM_DEVICE="cuda:0"
export SPEECHLLM_REPETITION_PENALTY="1.2"
export SPEECHLLM_REPETITION_PENALTY_WINDOW="8"

export SPEECHLLM_USE_VAD="1"
export SPEECHLLM_VAD_THRESHOLD="0.5"
export SPEECHLLM_VAD_MIN_SILENCE_DURATION_MS="250"
export SPEECHLLM_VAD_SPEECH_PAD_MS="150"

PORT="${PORT:-8001}"

echo "Root: $ROOT_DIR"
echo "Config: $SPEECHLLM_CONFIG"
echo "Checkpoint: $SPEECHLLM_CHECKPOINT"
echo "AST: $SPEECHLLM_AST_LANG_OPTIONS  ASR: $SPEECHLLM_ASR_LANG_OPTIONS"
echo "Serving on http://0.0.0.0:${PORT}"

uvicorn speechllm.app.stream_asr_ast_parallel.server:app --host 0.0.0.0 --port "$PORT"
