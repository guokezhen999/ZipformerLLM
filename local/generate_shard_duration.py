#!/usr/bin/env python3
"""
扫描 Shar 目录中所有 cuts.*.jsonl.gz，计算每个 shard 的总时长，
输出 JSON 供 assign_shards_to_ranks.py 使用。

Usage:
    python local/generate_shard_duration.py \
        --shar-dir data/fbank/shar/train \
        --output data/fbank/shar/train_shard_durations.json \
        --num-workers 16
"""

import argparse
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool


def compute_shard_duration(shard_path: str) -> dict:
    """计算单个 shard 的总时长（小时）。"""
    total_seconds = 0.0
    num_cuts = 0
    try:
        with gzip.open(shard_path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                total_seconds += obj.get("duration", 0.0)
                num_cuts += 1
    except Exception as e:
        print(f"[WARNING] Failed to read {shard_path}: {e}", file=sys.stderr)
        return {"path": os.path.basename(shard_path), "duration_hours": 0.0, "num_cuts": 0}

    return {
        "path": os.path.basename(shard_path),
        "duration_hours": round(total_seconds / 3600.0, 6),
        "num_cuts": num_cuts,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate shard duration JSON file")
    parser.add_argument(
        "--shar-dir",
        type=str,
        required=True,
        help="Path to Shar directory containing cuts.*.jsonl.gz files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    args = parser.parse_args()

    shard_files = sorted(glob.glob(os.path.join(args.shar_dir, "cuts.*.jsonl.gz")))
    if not shard_files:
        print(f"ERROR: No cuts.*.jsonl.gz files found in {args.shar_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shard files in {args.shar_dir}")
    print(f"Using {args.num_workers} workers...")

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(compute_shard_duration, shard_files)

    total_hours = sum(r["duration_hours"] for r in results)
    total_cuts = sum(r["num_cuts"] for r in results)

    output = {
        "shar_dir": args.shar_dir,
        "total_shards": len(results),
        "total_hours": round(total_hours, 4),
        "total_cuts": total_cuts,
        "shards": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(results)} shards, total {total_hours:.2f} hours, {total_cuts} cuts")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
