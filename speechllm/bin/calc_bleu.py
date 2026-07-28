import argparse
import json
import re
import sacrebleu


def remove_spaces_around_cjk(text):
    """Remove spaces between CJK characters and adjacent characters."""
    text = re.sub(r'(?<=\S) +(?=[一-鿿぀-ヿ])', '', text)
    text = re.sub(r'(?<=[一-鿿぀-ヿ]) +(?=\S)', '', text)
    return text


def load_references(ref_file, lang="zh"):
    """Build id -> {ref, src} mapping from jsonl."""
    refs = {}
    with open(ref_file) as f:
        for line in f:
            item = json.loads(line)
            uid = item["id"]
            for sup in item.get("supervisions", []):
                for trans in sup.get("translation", []):
                    if trans.get("lang") == lang:
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
    parser.add_argument("--lang", default="zh", help="Target language code for reference lookup")
    parser.add_argument("--tokenize", default="zh", help="sacrebleu tokenizer: zh, 13a, intl, char, none")
    args = parser.parse_args()

    refs = load_references(args.ref_file, lang=args.lang)

    hyps, ref_list, results = [], [], []
    skipped = []

    with open(args.hyp_file) as f:
        for line in f:
            item = json.loads(line)
            uid = item["id"]
            segments = item.get("segments_text", [])
            hyp = remove_spaces_around_cjk(" ".join(s for s in segments if s.strip()))

            if not hyp or uid not in refs:
                skipped.append({"id": uid, "hyp": hyp,
                                 "ref": refs.get(uid, {}).get("ref", ""),
                                 "src": refs.get(uid, {}).get("src", "")})
                continue

            hyps.append(hyp)
            ref_list.append(refs[uid]["ref"])
            results.append({
                "id": uid,
                "hyp": hyp,
                "ref": refs[uid]["ref"],
                "src": refs[uid]["src"],
            })

    if hyps:
        bleu = sacrebleu.corpus_bleu(hyps, [ref_list], tokenize=args.tokenize)
        avg_bleu = bleu.score
        bleu_str = bleu.format()
    else:
        avg_bleu = 0.0
        bleu_str = "N/A"

    with open(args.output_file, "w") as f:
        for r in results:
            sent_bleu = sacrebleu.sentence_bleu(r["hyp"], [r["ref"]], tokenize=args.tokenize).score
            f.write(f"ID:       {r['id']}\n")
            f.write(f"BLEU:     {sent_bleu:.2f}\n")
            f.write(f"原文:     {r['src']}\n")
            f.write(f"参考译文: {r['ref']}\n")
            f.write(f"实际译文: {r['hyp']}\n")
            f.write("-" * 80 + "\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total samples:   {len(results) + len(skipped)}\n")
        f.write(f"Skipped (empty): {len(skipped)}\n")
        f.write(f"Corpus BLEU:     {avg_bleu:.2f}\n")
        f.write(f"sacrebleu:       {bleu_str}\n")

    summary = (
        f"Total samples:   {len(results) + len(skipped)}\n"
        f"Skipped (empty): {len(skipped)}\n"
        f"Corpus BLEU:     {avg_bleu:.2f}\n"
        f"sacrebleu:       {bleu_str}\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
