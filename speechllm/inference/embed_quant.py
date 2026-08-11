"""Symmetric INT8 quantization helpers for SpeechLLM injected embeddings.

Used for:
  - special-token input patches (<A>, </A>)
  - audio embeddings from encoder+adapter

llama.cpp batch.embd still needs float32 at decode time, so runtime path is
quantize -> (optional store/transmit as int8) -> dequantize before prefill.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, float]


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def tensor_absmax_scale(x: np.ndarray, qmax: int = 127) -> float:
    """Per-tensor symmetric scale: absmax / qmax."""
    x = _as_f32(x)
    absmax = float(np.max(np.abs(x))) if x.size else 0.0
    if absmax <= 0.0:
        return 1.0
    return absmax / float(qmax)


def per_token_absmax_scale(x: np.ndarray, qmax: int = 127) -> np.ndarray:
    """Per-row (token) symmetric scales, shape (T, 1)."""
    x = _as_f32(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    absmax = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = absmax / float(qmax)
    return np.maximum(scale, np.float32(1e-8))


def quantize_symmetric_int8(
    x: np.ndarray,
    scale: ArrayLike | None = None,
    mode: str = "tensor",
    qmax: int = 127,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize to int8.

    Args:
        x: (D,) or (T, D)
        scale: optional precomputed scale(s). If None, computed from x.
        mode: "tensor" (one scale) or "per_token" (scale per row)
    Returns:
        q_int8, scale (float32 scalar or (T,1))
    """
    x = _as_f32(x)
    squeeze = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True

    if scale is None:
        if mode == "per_token":
            scale_arr = per_token_absmax_scale(x, qmax=qmax)
        elif mode == "tensor":
            scale_arr = np.asarray(tensor_absmax_scale(x, qmax=qmax), dtype=np.float32)
        else:
            raise ValueError(f"Unknown quant mode: {mode}")
    else:
        scale_arr = np.asarray(scale, dtype=np.float32)

    q = np.clip(np.round(x / scale_arr), -128, 127).astype(np.int8)
    if squeeze and mode == "tensor":
        return q.reshape(-1), scale_arr
    if squeeze and mode == "per_token":
        return q.reshape(-1), scale_arr.reshape(())
    return q, scale_arr


def dequantize_symmetric_int8(q: np.ndarray, scale: ArrayLike) -> np.ndarray:
    """Dequantize int8 embeddings with broadcastable scale."""
    q = np.asarray(q)
    scale_arr = np.asarray(scale, dtype=np.float32)
    return q.astype(np.float32) * scale_arr


def quantize_dequantize(
    x: np.ndarray,
    mode: str = "per_token",
    scale: ArrayLike | None = None,
    qmax: int = 127,
) -> np.ndarray:
    """INT8 round-trip (simulates quantized audio/special-token embeddings)."""
    q, used_scale = quantize_symmetric_int8(x, scale=scale, mode=mode, qmax=qmax)
    y = dequantize_symmetric_int8(q, used_scale)
    return y.astype(np.float32)


def pack_special_token_int8(
    emb_a: np.ndarray,
    emb_a_end: np.ndarray,
    emb_w: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Pack <A>/</A>/(optional <W>) as int8 + per-vector scales."""
    qa, sa = quantize_symmetric_int8(emb_a, mode="tensor")
    qe, se = quantize_symmetric_int8(emb_a_end, mode="tensor")
    out: Dict[str, np.ndarray] = {
        "emb_a": qa,
        "emb_a_scale": np.asarray(sa, dtype=np.float32),
        "emb_a_end": qe,
        "emb_a_end_scale": np.asarray(se, dtype=np.float32),
        "quant_scheme": np.asarray("symmetric_int8_per_vector"),
    }
    if emb_w is not None:
        qw, sw = quantize_symmetric_int8(emb_w, mode="tensor")
        out["emb_w"] = qw
        out["emb_w_scale"] = np.asarray(sw, dtype=np.float32)
    return out


def unpack_special_token_int8(data) -> Tuple[np.ndarray, np.ndarray]:
    """Load float <A>/</A> from either float npz or int8+scale npz."""
    files = set(getattr(data, "files", data.keys()))
    if "emb_a_scale" in files:
        emb_a = dequantize_symmetric_int8(data["emb_a"], data["emb_a_scale"])
        emb_a_end = dequantize_symmetric_int8(data["emb_a_end"], data["emb_a_end_scale"])
        return emb_a.astype(np.float32), emb_a_end.astype(np.float32)
    return np.asarray(data["emb_a"], dtype=np.float32), np.asarray(data["emb_a_end"], dtype=np.float32)
