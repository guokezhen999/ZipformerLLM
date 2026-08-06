#!/usr/bin/env python3
"""Evaluate streaming AST latency metrics from decode JSONL.

Metrics (skip = utterances with no non-empty chunk; excluded from latency avgs):
  - First-token latency (FTL, seconds)
  - Write count (non-empty chunks per utterance)
  - Wait ratio

Also reports MuST-COMMON + MuST-HE weighted averages (by sample counts),
grouped by chunk size.

FTL time: end_idx * FRAME_SEC + OFFSET_SEC

Usage:
  python calc_emission_latency.py [dir1 dir2 ...]
  python calc_emission_latency.py path/to/file.jsonl
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FRAME_SEC = 0.16
OFFSET_SEC = 0.025

DEFAULT_DIRS = [
    "/nfs_tmk/asr/guokezhen/SpeechLLM/examples/ast_en_zh_20260609/exp/sft_ast_stage2/decode_ast_must",
    "/nfs_tmk/asr/guokezhen/SpeechLLM/examples/ast_en_zh_20260609/exp/grpo_v1/decode_comet_ast_must",
    "/nfs_tmk/asr/guokezhen/SpeechLLM/examples/ast_en_zh_20260609/exp/grpo_v1/decode_bleu_ast_must",
]

REPORT_NAME = "emission_latency_write_report.txt"

DATASET_RE = re.compile(r"(MuST-COMMON|MuST-HE)", re.IGNORECASE)
CHUNK_RE = re.compile(r"chunk_(\d+)", re.IGNORECASE)


def chunk_emit_time(end_idx: float | int) -> float:
    return float(end_idx) * FRAME_SEC + OFFSET_SEC


def is_write_chunk(chunk: dict) -> bool:
    text = chunk.get("text") or ""
    return bool(str(text).strip())


def parse_meta(name: str) -> tuple[str | None, str | None]:
    """Return (dataset, chunk_tag) from filename, e.g. ('MuST-COMMON', 'chunk_128')."""
    ds_m = DATASET_RE.search(name)
    ch_m = CHUNK_RE.search(name)
    dataset = None
    if ds_m:
        raw = ds_m.group(1).upper()
        dataset = "MuST-COMMON" if "COMMON" in raw else "MuST-HE"
    chunk = f"chunk_{ch_m.group(1)}" if ch_m else None
    return dataset, chunk


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def eval_jsonl(jsonl_path: str | Path) -> dict | None:
    """Compute metrics for one JSONL file."""
    path = Path(jsonl_path)
    dataset, chunk = parse_meta(path.name)

    total = 0
    skipped = 0

    utt_ftl: list[float] = []
    utt_write_counts: list[int] = []
    utt_total_chunks: list[int] = []
    utt_wait_ratios: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            chunks = obj.get("chunks") or []
            n_chunks = len(chunks)
            writes = [c for c in chunks if is_write_chunk(c)]
            n_writes = len(writes)

            utt_total_chunks.append(n_chunks)
            utt_write_counts.append(n_writes)
            if n_chunks > 0:
                utt_wait_ratios.append((n_chunks - n_writes) / n_chunks)

            if not writes:
                skipped += 1
                continue

            # first-token latency: first write chunk only
            utt_ftl.append(chunk_emit_time(writes[0]["end_idx"]))

    if total == 0:
        return None

    return {
        "path": str(path),
        "name": path.name,
        "dataset": dataset,
        "chunk": chunk,
        "total": total,
        "with_output": total - skipped,
        "skipped": skipped,
        # first-token latency (skip excluded)
        "avg_ftl": _mean(utt_ftl),
        "med_ftl": _median(utt_ftl),
        "min_ftl": min(utt_ftl) if utt_ftl else None,
        "max_ftl": max(utt_ftl) if utt_ftl else None,
        # write / wait
        "avg_writes": _mean([float(x) for x in utt_write_counts]),
        "med_writes": _median([float(x) for x in utt_write_counts]),
        "min_writes": min(utt_write_counts) if utt_write_counts else None,
        "max_writes": max(utt_write_counts) if utt_write_counts else None,
        "avg_chunks": _mean([float(x) for x in utt_total_chunks]),
        "avg_wait_ratio": _mean(utt_wait_ratios),
    }


def _wavg(pairs: list[tuple[float, float]]) -> float | None:
    """Weighted average of (value, weight) pairs."""
    pairs = [(v, w) for v, w in pairs if v is not None and w and w > 0]
    if not pairs:
        return None
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    return num / den if den else None


def weighted_common_he(results: list[dict]) -> list[dict]:
    """
    For each chunk_* group that has both MuST-COMMON and MuST-HE,
    compute weighted averages.

    Latency metrics (FTL): weight by with_output (skip excluded).
    Write / wait / chunks: weight by total samples.
    """
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        if r.get("dataset") and r.get("chunk"):
            groups[r["chunk"]][r["dataset"]] = r

    out: list[dict] = []
    for chunk in sorted(groups.keys(), key=lambda x: int(x.split("_")[1])):
        g = groups[chunk]
        common = g.get("MuST-COMMON")
        he = g.get("MuST-HE")
        if not common or not he:
            continue

        w_out = [
            (common["avg_ftl"], common["with_output"]),
            (he["avg_ftl"], he["with_output"]),
        ]
        w_writes = [
            (common["avg_writes"], common["total"]),
            (he["avg_writes"], he["total"]),
        ]
        w_chunks = [
            (common["avg_chunks"], common["total"]),
            (he["avg_chunks"], he["total"]),
        ]
        w_wait = [
            (common["avg_wait_ratio"], common["total"]),
            (he["avg_wait_ratio"], he["total"]),
        ]

        out.append(
            {
                "name": f"WEIGHTED(COMMON+HE)_{chunk}",
                "chunk": chunk,
                "total": common["total"] + he["total"],
                "with_output": common["with_output"] + he["with_output"],
                "skipped": common["skipped"] + he["skipped"],
                "n_common": common["total"],
                "n_he": he["total"],
                "out_common": common["with_output"],
                "out_he": he["with_output"],
                "avg_ftl": _wavg(w_out),
                "avg_writes": _wavg(w_writes),
                "avg_chunks": _wavg(w_chunks),
                "avg_wait_ratio": _wavg(w_wait),
            }
        )
    return out


def _fmt(v: float | None, nd: int = 4) -> str:
    return f"{v:.{nd}f}" if v is not None else "NA"


def format_report(dir_path: Path, results: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"Directory: {dir_path}")
    lines.append(
        f"Time formula: end_idx * {FRAME_SEC} + {OFFSET_SEC} (seconds)"
    )
    lines.append(
        "FTL (first-token latency): emit time of first non-empty chunk; "
        "skip (no write) excluded from average"
    )
    lines.append(
        "writes / med_w: mean / median Write count (# non-empty chunks per utt)"
    )
    lines.append(
        "Weighted COMMON+HE: FTL by with_output; writes/wait/chunks by N"
    )
    lines.append("")

    header = (
        f"{'file':<62} "
        f"{'N':>5} {'out':>5} {'skip':>4} "
        f"{'FTL':>8} "
        f"{'writes':>7} {'med_w':>6} "
        f"{'wait%':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        wait = (
            f"{100.0 * r['avg_wait_ratio']:.1f}"
            if r.get("avg_wait_ratio") is not None
            else "NA"
        )
        med_w = (
            f"{r['med_writes']:.1f}"
            if r.get("med_writes") is not None
            else "NA"
        )
        lines.append(
            f"{r['name']:<62} "
            f"{r['total']:>5} {r['with_output']:>5} {r['skipped']:>4} "
            f"{_fmt(r.get('avg_ftl')):>8} "
            f"{_fmt(r.get('avg_writes'), 3):>7} "
            f"{med_w:>6} "
            f"{wait:>6}"
        )

    lines.append("-" * len(header))
    lines.append("")

    # Weighted COMMON + HE
    weighted = weighted_common_he(results)
    lines.append("=== Weighted average: MuST-COMMON + MuST-HE (by sample counts) ===")
    lines.append("")
    if not weighted:
        lines.append("(no COMMON/HE pair found for any chunk size)")
        lines.append("")
    else:
        wh = (
            f"{'group':<28} "
            f"{'N':>5} {'out':>5} "
            f"{'N_C':>5} {'N_H':>5} "
            f"{'FTL':>8} "
            f"{'writes':>7} {'wait%':>6}"
        )
        lines.append(wh)
        lines.append("-" * len(wh))
        for w in weighted:
            wait = (
                f"{100.0 * w['avg_wait_ratio']:.1f}"
                if w["avg_wait_ratio"] is not None
                else "NA"
            )
            lines.append(
                f"{w['name']:<28} "
                f"{w['total']:>5} {w['with_output']:>5} "
                f"{w['n_common']:>5} {w['n_he']:>5} "
                f"{_fmt(w['avg_ftl']):>8} "
                f"{_fmt(w['avg_writes'], 3):>7} "
                f"{wait:>6}"
            )
        lines.append("-" * len(wh))
        lines.append("")

    lines.append("Per-file details:")
    lines.append("")

    for r in results:
        lines.append(f"[{r['name']}]")
        lines.append(f"  dataset / chunk         : {r.get('dataset')} / {r.get('chunk')}")
        lines.append(f"  samples                 : {r['total']}")
        lines.append(f"  with output             : {r['with_output']}")
        lines.append(f"  skipped (no write)      : {r['skipped']}")
        if r.get("avg_ftl") is not None:
            lines.append(
                f"  avg first-token latency : {r['avg_ftl']:.4f}s "
                f"(med={r['med_ftl']:.4f}, "
                f"min={r['min_ftl']:.4f}, max={r['max_ftl']:.4f})"
            )
            lines.append(
                f"  avg Write count          : {r['avg_writes']:.4f} "
                f"(med={r['med_writes']:.1f}, "
                f"min={r['min_writes']}, max={r['max_writes']})"
            )
            lines.append(
                f"  avg total chunks         : {r['avg_chunks']:.4f}"
            )
            lines.append(
                f"  avg Wait ratio           : {100.0 * r['avg_wait_ratio']:.2f}%"
            )
        else:
            lines.append("  (no write chunks)")
        lines.append("")

    if weighted:
        lines.append("Weighted COMMON+HE details:")
        lines.append("")
        for w in weighted:
            lines.append(f"[{w['name']}]")
            lines.append(
                f"  N=COMMON({w['n_common']})+HE({w['n_he']})={w['total']}, "
                f"out=COMMON({w['out_common']})+HE({w['out_he']})={w['with_output']}, "
                f"skip={w['skipped']}"
            )
            lines.append(f"  FTL                      : {_fmt(w['avg_ftl'])}s")
            lines.append(f"  writes                   : {_fmt(w['avg_writes'], 4)}")
            lines.append(f"  chunks                   : {_fmt(w['avg_chunks'], 4)}")
            wait = (
                f"{100.0 * w['avg_wait_ratio']:.2f}%"
                if w["avg_wait_ratio"] is not None
                else "NA"
            )
            lines.append(f"  wait%                    : {wait}")
            lines.append("")

    return "\n".join(lines) + "\n"


def process_directory(dir_path: str | Path, write_report: bool = True) -> Path | None:
    d = Path(dir_path)
    if not d.is_dir():
        print(f"[skip] not a directory: {d}")
        return None

    jsonl_files = sorted(d.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[skip] no jsonl in: {d}")
        return None

    results = []
    for f in jsonl_files:
        r = eval_jsonl(f)
        if r is not None:
            results.append(r)

    if not results:
        print(f"[skip] no valid results in: {d}")
        return None

    report = format_report(d, results)
    print(report)

    if write_report:
        out = d / REPORT_NAME
        out.write_text(report, encoding="utf-8")
        print(f"[saved] {out}\n")
        return out
    return None


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else DEFAULT_DIRS

    if len(targets) == 1 and Path(targets[0]).is_file():
        r = eval_jsonl(targets[0])
        if r is None:
            print("No valid samples.")
            return
        print(format_report(Path(targets[0]).parent, [r]))
        return

    for t in targets:
        process_directory(t, write_report=True)


if __name__ == "__main__":
    main(sys.argv)
