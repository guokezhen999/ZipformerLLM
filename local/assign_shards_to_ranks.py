#!/usr/bin/env python3
"""
根据 shard 时长文件，用 LPT 贪心将 shard 均衡分配给各 rank。

LPT (Longest Processing Time first)：
  1. 按时长降序排列所有 shard
  2. 依次将最长的 shard 分给当前总时长最短的 rank

Usage:
    python local/assign_shards_to_ranks.py \
        --duration-file data/fbank/shar/train_shard_durations.json \
        --world-size 16 \
        --output exp/run/shard_assignment_ws16.json
"""

import argparse
import heapq
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Assign shards to ranks using LPT algorithm")
    parser.add_argument(
        "--duration-file",
        type=str,
        required=True,
        help="Path to shard_durations.json from generate_shard_duration.py",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="Number of ranks (GPUs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output assignment JSON file path",
    )
    args = parser.parse_args()

    with open(args.duration_file, "r") as f:
        duration_data = json.load(f)

    shards = duration_data["shards"]
    shar_dir = duration_data["shar_dir"]
    world_size = args.world_size

    if world_size <= 0:
        print("ERROR: world-size must be positive", file=sys.stderr)
        sys.exit(1)

    if len(shards) < world_size:
        print(
            f"WARNING: only {len(shards)} shards for {world_size} ranks, "
            f"some ranks will have no data",
            file=sys.stderr,
        )

    # LPT: 按时长降序排列
    shards_sorted = sorted(shards, key=lambda s: s["duration_hours"], reverse=True)

    # 最小堆: (当前总时长, rank_id)
    heap = [(0.0, r) for r in range(world_size)]
    heapq.heapify(heap)

    assignment = {str(r): [] for r in range(world_size)}

    for shard in shards_sorted:
        total_hours, rank_id = heapq.heappop(heap)
        assignment[str(rank_id)].append(shard["path"])
        heapq.heappush(heap, (total_hours + shard["duration_hours"], rank_id))

    shard_dur_map = {s["path"]: s["duration_hours"] for s in shards}
    rank_total_hours = {}
    for r in range(world_size):
        total = sum(shard_dur_map.get(p, 0.0) for p in assignment[str(r)])
        rank_total_hours[str(r)] = round(total, 4)

    hours_list = list(rank_total_hours.values())
    max_h = max(hours_list) if hours_list else 0
    min_h = min(hours_list) if hours_list else 0
    avg_h = sum(hours_list) / len(hours_list) if hours_list else 0

    output = {
        "shar_dir": shar_dir,
        "world_size": world_size,
        "total_shards": len(shards),
        "assignment": assignment,
        "rank_total_hours": rank_total_hours,
        "stats": {
            "max_hours": round(max_h, 4),
            "min_hours": round(min_h, 4),
            "avg_hours": round(avg_h, 4),
            "imbalance_pct": round((max_h - min_h) / avg_h * 100, 2) if avg_h > 0 else 0,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Assignment complete: {len(shards)} shards -> {world_size} ranks")
    print(f"  Avg: {avg_h:.2f}h, Min: {min_h:.2f}h, Max: {max_h:.2f}h")
    print(f"  Imbalance: {output['stats']['imbalance_pct']:.2f}%")
    for r in range(min(world_size, 4)):
        print(f"  Rank {r}: {len(assignment[str(r)])} shards, {rank_total_hours[str(r)]:.2f}h")
    if world_size > 4:
        print(f"  ... ({world_size - 4} more ranks)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
