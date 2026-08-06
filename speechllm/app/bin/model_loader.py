"""
Shared model loading utilities for all streaming server variants.
"""
import os
import logging
import torch
from speechllm.streaming_model import SpeechLLMASRStream
from speechllm.utils import load_config
from addict import Dict

if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([Dict])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(
    config_path: str,
    ckpt_path: str,
    device_str: str = None,
    tag: str = "",
    llm_path: str = None,
    load_llm_from_checkpoint: bool = None,
):
    """Load a SpeechLLM model from config and checkpoint paths.

    When ``llm_path`` is set (HF export dir), it overrides ``config.model.llm.model_name``
    and checkpoint LLM/patch weights are skipped by default.

    Returns:
        tuple: (model, config, device)
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = f"[{tag}] " if tag else ""
    logging.info(f"{prefix}Loading model from {ckpt_path} ...")
    config = load_config(config_path)

    if llm_path:
        if not hasattr(config, "model") or config.model is None:
            config.model = Dict()
        if not hasattr(config.model, "llm") or config.model.llm is None:
            config.model.llm = Dict()
        config.model.llm.model_name = llm_path
        logging.info(f"{prefix}LLM path from env/arg: {llm_path}")
        if load_llm_from_checkpoint is None:
            load_llm_from_checkpoint = False
    elif load_llm_from_checkpoint is None:
        load_llm_from_checkpoint = True

    if not config.model.llm.get("model_name"):
        raise ValueError(
            "LLM path missing: set SPEECHLLM_LLM_PATH or config.model.llm.model_name"
        )

    device = torch.device(device_str)
    model = SpeechLLMASRStream(config, device)
    model.load_checkpoint(ckpt_path, load_llm=load_llm_from_checkpoint)
    model.eval()
    logging.info(f"{prefix}Model loaded on {device}.")
    return model, config, device


def load_model_from_env(
    config_env="SPEECHLLM_CONFIG",
    ckpt_env="SPEECHLLM_CHECKPOINT",
    device_env="SPEECHLLM_DEVICE",
    llm_env="SPEECHLLM_LLM_PATH",
    tag: str = "",
):
    """Load model using environment variables. Returns (model, config, device) or (None, None, None).

    ``SPEECHLLM_LLM_PATH`` (HF LLM dir) overrides config.model.llm.model_name when set.
    Deploy configs without model_name require this env var.
    """
    config_path = os.environ.get(config_env)
    ckpt_path = os.environ.get(ckpt_env)
    if not config_path or not ckpt_path:
        logging.error(f"Missing {config_env} or {ckpt_env} env vars")
        return None, None, None
    device_str = os.environ.get(device_env, "cuda" if torch.cuda.is_available() else "cpu")
    llm_path = os.environ.get(llm_env) or None
    try:
        return load_model(config_path, ckpt_path, device_str, tag, llm_path=llm_path)
    except ValueError as e:
        logging.error(str(e))
        return None, None, None
