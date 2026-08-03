#!/usr/bin/env python3
"""Extract model weights only from a Lightning / GRPO RL checkpoint.

Supports:
  - DeepSpeed Lightning ckpt dir, e.g. step-step=1000.ckpt/
  - DeepSpeed mp_rank_00_model_states.pt
  - Single-file Lightning .ckpt with state_dict

Usage:
  python speechllm/utils/convert_rl_ckpt.py \\
    exp/.../checkpoints/step-step=1000.ckpt \\
    exp/.../checkpoints/step-1000-weights.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# PyTorch 2.6+: torch.load defaults to weights_only=True, breaks DS ckpt loading.
_original_load = torch.load


def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load


def resolve_model_states_path(path: Path) -> Path:
    """Map ckpt dir / file to the pt file that holds weights."""
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Input checkpoint not found: {path}")

    candidate = path / "checkpoint" / "mp_rank_00_model_states.pt"
    if candidate.is_file():
        return candidate

    matches = sorted(path.glob("**/mp_rank_00_model_states.pt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No mp_rank_00_model_states.pt under DeepSpeed ckpt dir: {path}"
    )


def extract_state_dict(ckpt: dict) -> dict:
    if "module" in ckpt and isinstance(ckpt["module"], dict):
        return ckpt["module"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]

    tensor_like = sum(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:32])
    if tensor_like > 0:
        return ckpt

    raise KeyError(
        f"No weights found. Top-level keys: {sorted(ckpt.keys())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RL/Lightning DeepSpeed ckpt to weights-only .pt"
    )
    parser.add_argument(
        "input_ckpt",
        type=str,
        help="DeepSpeed ckpt dir (*.ckpt/) or a .pt/.ckpt file",
    )
    parser.add_argument(
        "output_ckpt",
        type=str,
        help="Output path for weights-only checkpoint",
    )
    args = parser.parse_args()

    in_path = Path(args.input_ckpt).expanduser().resolve()
    out_path = Path(args.output_ckpt).expanduser().resolve()
    model_states = resolve_model_states_path(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {model_states}")
    ckpt = torch.load(str(model_states), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    state_dict = extract_state_dict(ckpt)
    print(f"Extracted {len(state_dict)} tensors")

    torch.save({"state_dict": state_dict}, str(out_path))
    size_gb = out_path.stat().st_size / (1024**3)
    print(f"Saved weights-only ckpt: {out_path} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
