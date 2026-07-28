"""共享内存 Embedding 传输层。

训练进程将 embedding tensor 写入 /dev/shm，本地代理进程从共享内存读取
并转发给 SGLang server，避免在训练进程中做昂贵的 .tolist() + JSON 序列化。

架构：
  Trainer  ──(shm write + lightweight HTTP)──>  ShmProxy  ──(JSON)──>  SGLang
           <──────── result JSON ──────────────            <──────────

使用方式：
  1. 启动代理：  ShmProxyServer(sglang_url, proxy_port).start()  (独立进程)
  2. 训练端调用：ShmTransportClient(proxy_port).send(embeds_tensor, sampling_params)
"""

import json
import logging
import mmap
import os
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 共享内存目录，Linux 下 /dev/shm 是 tmpfs，零磁盘 IO
SHM_DIR = Path("/dev/shm/sglang_shm_transport")

# 头部格式：ndim(4B) + shape(ndim * 8B) + dtype_len(4B) + dtype_str
HEADER_STRUCT_PREFIX = "<I"  # ndim as uint32


def _ensure_shm_dir():
    SHM_DIR.mkdir(parents=True, exist_ok=True)


def write_tensor_to_shm(tensor, dtype_str: str = "float32") -> Tuple[str, list, str]:
    """将 tensor 写入 /dev/shm，返回 (shm_path, shape, dtype_str)。

    Args:
        tensor: PyTorch tensor 或 numpy array
        dtype_str: 存储精度，"float32" 或 "float16"/"bfloat16"

    Returns:
        (shm_path, shape_list, dtype_str)
    """
    _ensure_shm_dir()

    # 转 numpy
    if hasattr(tensor, "cpu"):
        # PyTorch tensor
        if dtype_str == "float16":
            arr = tensor.half().cpu().numpy()
        else:
            arr = tensor.float().cpu().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32 if dtype_str == "float32" else np.float16)

    shape = list(arr.shape)
    uid = uuid.uuid4().hex[:12]
    shm_path = str(SHM_DIR / f"emb_{uid}.bin")

    # 直接写 raw bytes，比任何序列化都快
    arr.tofile(shm_path)

    return shm_path, shape, str(arr.dtype)


def read_tensor_from_shm(shm_path: str, shape: list, dtype_str: str) -> np.ndarray:
    """从 /dev/shm 读取 tensor。

    Args:
        shm_path: 共享内存文件路径
        shape: tensor shape
        dtype_str: numpy dtype 字符串

    Returns:
        numpy ndarray
    """
    arr = np.fromfile(shm_path, dtype=np.dtype(dtype_str))
    arr = arr.reshape(shape)
    return arr


def cleanup_shm_file(shm_path: str):
    """清理共享内存文件。"""
    try:
        if os.path.exists(shm_path):
            os.unlink(shm_path)
    except OSError:
        pass


def cleanup_all_shm():
    """清理所有共享内存文件。"""
    if SHM_DIR.exists():
        for f in SHM_DIR.glob("emb_*.bin"):
            try:
                f.unlink()
            except OSError:
                pass
