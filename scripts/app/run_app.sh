#!/bin/bash

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

conda activate speechllm

export SPEECHLLM_AST_LANG_OPTIONS="Chinese"
export SPEECHLLM_ASR_LANG_OPTIONS="English"

export SPEECHLLM_CONFIG="conf/deploy_model.json"
export SPEECHLLM_CHECKPOINT="exp/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000/checkpoints/best-step-step=1950.pt"
export SPEECHLLM_LLM_PATH="pretrained_models/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000_llm"
# export SPEECHLLM_CHECKPOINT="exp/grpo_vllm_kl_0.05_lr_1e-6_comet_no_kl_step_2000/checkpoints/step-step=2000.ckpt"
export SPEECHLLM_DEVICE="cuda:0"
export SPEECHLLM_REPETITION_PENALTY="1.0"
export SPEECHLLM_REPETITION_PENALTY_WINDOW="8"
export SPEECHLLM_KEEP_SEGMENTS="16"
# export SPEECHLLM_MAX_SEGMENTS="64"

export SPEECHLLM_USE_VAD="1"
export SPEECHLLM_VAD_THRESHOLD="0.5"
export SPEECHLLM_VAD_MIN_SILENCE_DURATION_MS="250"
export SPEECHLLM_VAD_SPEECH_PAD_MS="150"

echo "AST: $SPEECHLLM_AST_LANG_OPTIONS  ASR: $SPEECHLLM_ASR_LANG_OPTIONS"
echo "Config: $SPEECHLLM_CONFIG  Checkpoint: $SPEECHLLM_CHECKPOINT  LLM: $SPEECHLLM_LLM_PATH"

uvicorn speechllm.app.stream_asr_ast_parallel.server:app --host 0.0.0.0 --port 8001
