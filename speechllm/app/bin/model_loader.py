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


def load_model(config_path: str, ckpt_path: str, device_str: str = None, tag: str = ""):
    """Load a SpeechLLM model from config and checkpoint paths.

    Returns:
        tuple: (model, config, device)
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = f"[{tag}] " if tag else ""
    logging.info(f"{prefix}Loading model from {ckpt_path} ...")
    config = load_config(config_path)
    device = torch.device(device_str)
    model = SpeechLLMASRStream(config, device)
    model.load_checkpoint(ckpt_path)
    model.eval()
    logging.info(f"{prefix}Model loaded on {device}.")
    return model, config, device


def load_model_from_env(
    config_env="SPEECHLLM_CONFIG",
    ckpt_env="SPEECHLLM_CHECKPOINT",
    device_env="SPEECHLLM_DEVICE",
    tag: str = "",
):
    """Load model using environment variables. Returns (model, config, device) or (None, None, None)."""
    config_path = os.environ.get(config_env)
    ckpt_path = os.environ.get(ckpt_env)
    if not config_path or not ckpt_path:
        logging.error(f"Missing {config_env} or {ckpt_env} env vars")
        return None, None, None
    device_str = os.environ.get(device_env, "cuda" if torch.cuda.is_available() else "cpu")
    return load_model(config_path, ckpt_path, device_str, tag)
