#!/usr/bin/env python3
"""Convert Lightning DeepSpeed ZeRO checkpoints to a single fp32 .pt file.

Usage:
  python speechllm/utils/convert_ds_ckpt.py /path/to/step-xxx.ckpt
  python speechllm/utils/convert_ds_ckpt.py /path/to/step-xxx.ckpt -o /path/to/out.pt
  python speechllm/utils/convert_ds_ckpt.py /path/to/checkpoints --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# PyTorch 2.6+: torch.load defaults to weights_only=True, which breaks DeepSpeed ckpt loading.
_original_load = torch.load


def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def default_output_path(ckpt_dir: Path) -> Path:
    """Derive output .pt path from a DeepSpeed checkpoint directory."""
    name = ckpt_dir.name
    if name.endswith(".ckpt"):
        return ckpt_dir.with_suffix(".pt")
    return ckpt_dir.parent / f"{name}.pt"


def is_deepspeed_ckpt_dir(path: Path) -> bool:
    """Heuristic: Lightning DeepSpeed ckpt dirs contain checkpoint/ or zero_* files."""
    if not path.is_dir():
        return False
    if (path / "checkpoint").is_dir():
        return True
    return any(path.glob("zero_*")) or (path / "latest").exists()


def find_ckpt_dirs(root: Path) -> list[Path]:
    """Find DeepSpeed checkpoint dirs under root (non-recursive for *.ckpt, else one level)."""
    if is_deepspeed_ckpt_dir(root):
        return [root]

    candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p.name.endswith(".ckpt") or is_deepspeed_ckpt_dir(p)))
    return [p for p in candidates if is_deepspeed_ckpt_dir(p)]


def convert_one(ckpt_dir: Path, output_path: Path | None = None, overwrite: bool = False) -> Path:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a DeepSpeed checkpoint directory, got a file: {ckpt_dir}"
        )
    if not is_deepspeed_ckpt_dir(ckpt_dir):
        raise ValueError(
            f"Does not look like a DeepSpeed ZeRO checkpoint directory: {ckpt_dir}"
        )

    out = output_path or default_output_path(ckpt_dir)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {out}")

    logger.info("Converting: %s -> %s", ckpt_dir, out)
    convert_zero_checkpoint_to_fp32_state_dict(str(ckpt_dir), str(out))
    logger.info("Saved: %s (%.2f GB)", out, out.stat().st_size / (1024**3))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Lightning DeepSpeed ZeRO checkpoints to fp32 .pt state dicts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ckpt",
        type=str,
        help="DeepSpeed checkpoint directory (e.g. step-step=300000.ckpt), "
        "or a parent directory when used with --all",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output .pt path. Default: same path with .ckpt -> .pt "
        "(or <name>.pt if no .ckpt suffix). Ignored with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all DeepSpeed checkpoint dirs under the given path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output .pt files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt_root = Path(args.ckpt).expanduser().resolve()

    try:
        if args.all:
            ckpt_dirs = find_ckpt_dirs(ckpt_root)
            if not ckpt_dirs:
                logger.error("No DeepSpeed checkpoint directories found under: %s", ckpt_root)
                return 1
            if args.output is not None:
                logger.warning("--output is ignored when using --all")
            logger.info("Found %d checkpoint(s) to convert", len(ckpt_dirs))
            for ckpt_dir in ckpt_dirs:
                convert_one(ckpt_dir, overwrite=args.overwrite)
        else:
            output = Path(args.output).expanduser().resolve() if args.output else None
            convert_one(ckpt_root, output_path=output, overwrite=args.overwrite)
    except (FileNotFoundError, NotADirectoryError, ValueError, FileExistsError) as e:
        logger.error("%s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
