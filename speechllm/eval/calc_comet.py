import argparse
import json
import os
import re
from comet import load_from_checkpoint


def remove_spaces_around_cjk(text):
    """Remove spaces between CJK characters, full-width punctuation, and adjacent characters."""
    # CJK characters + CJK punctuation + full-width forms
    cjk_and_punct = (
        r'\u4e00-\u9fff'       # CJK Unified Ideographs
        r'\u3040-\u30ff'       # Hiragana + Katakana
        r'\u3000-\u303f'       # CJK Symbols and Punctuation (\u3002\u3001\u3003 etc.)
        r'\uff00-\uff5e'       # Fullwidth forms (\uff01\uff0c\uff0e\uff1a\uff1b\uff1f etc.)
        r'\u2018\u2019'        # Single curly quotes ''
        r'\u201c\u201d'        # Double curly quotes ""
        r'\u2026'              # Ellipsis \u2026
        r'\u2014'              # Em dash \u2014
    )
    text = re.sub(rf'(?<=\S) +(?=[{cjk_and_punct}])', '', text)
    text = re.sub(rf'(?<=[{cjk_and_punct}]) +(?=\S)', '', text)
    return text


def load_references(ref_file, ref_lang="zh"):
    """Build id -> reference translation mapping from jsonl."""
    refs = {}
    with open(ref_file) as f:
        for line in f:
            item = json.loads(line)
            uid = item["id"]
            for sup in item.get("supervisions", []):
                for trans in sup.get("translation", []):
                    if trans.get("lang") == ref_lang:
                        refs[uid] = {
                            "ref": trans["text"],
                            "src": sup.get("text", ""),
                        }
    return refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", required=True, help="Hypothesis jsonl file")
    parser.add_argument("--ref_file", required=True, help="Reference jsonl file")
    parser.add_argument("--output_file", required=True, help="Output txt file")
    parser.add_argument("--model", required=True, help="Local COMET model path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ref_lang", default="zh", help="Language code of reference translation (e.g. zh, en)")
    args = parser.parse_args()

    refs = load_references(args.ref_file, args.ref_lang)
    model = load_from_checkpoint(args.model, reload_hparams=True, local_files_only=True)

    samples, ids, hyps = [], [], []
    skipped = []

    with open(args.hyp_file) as f:
        for line in f:
            item = json.loads(line)
            uid = item["id"]
            segments = item.get("segments_text", [])
            hyp = remove_spaces_around_cjk(" ".join(s for s in segments if s.strip()))

            if not hyp or uid not in refs:
                skipped.append({"id": uid, "hyp": hyp, "comet_score": None,
                                 "ref": refs.get(uid, {}).get("ref", ""),
                                 "src": refs.get(uid, {}).get("src", "")})
                continue

            ids.append(uid)
            hyps.append(hyp)
            samples.append({
                "src": refs[uid]["src"],
                "mt": hyp,
                "ref": refs[uid]["ref"],
            })

    scores = []
    if samples:
        output = model.predict(samples, batch_size=args.batch_size, gpus=1)
        scores = output.scores

    results = []
    for uid, hyp, sample, score in zip(ids, hyps, samples, scores):
        results.append({
            "id": uid,
            "comet_score": score,
            "hyp": hyp,
            "src": sample["src"],
            "ref": sample["ref"],
        })

    results.extend(skipped)
    results.sort(key=lambda x: (x["comet_score"] is None, x["comet_score"] if x["comet_score"] is not None else 0))

    valid_scores = [r["comet_score"] for r in results if r["comet_score"] is not None]
    avg_comet = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    summary = (
        f"Total samples:   {len(results)}\n"
        f"Skipped (empty): {len(skipped)}\n"
        f"Average COMET:   {avg_comet:.4f}\n"
    )
    print(summary)

    with open(args.output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(summary)
        f.write("\n")
        for rank, r in enumerate(results, 1):
            score_str = f"{r['comet_score']:.4f}" if r["comet_score"] is not None else "N/A"
            f.write(f"Rank:     {rank}\n")
            f.write(f"ID:       {r['id']}\n")
            f.write(f"COMET:    {score_str}\n")
            f.write(f"原文:     {r['src']}\n")
            f.write(f"参考译文: {r['ref']}\n")
            f.write(f"实际译文: {r['hyp']}\n")
            f.write("-" * 80 + "\n")


if __name__ == "__main__":
    main()
