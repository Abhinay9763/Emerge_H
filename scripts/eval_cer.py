import argparse
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def parse_pairs(path: Path):
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            gt, pred = line.split("\t", maxsplit=1)
            pairs.append((gt, pred))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CER and WER from a TSV file: gt<TAB>pred"
    )
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    if not pairs:
        raise ValueError("No valid rows in pairs file")

    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0

    for gt, pred in pairs:
        total_char_dist += edit_distance(gt, pred)
        total_chars += max(1, len(gt))

        gt_words = gt.split()
        pred_words = pred.split()
        total_word_dist += edit_distance("\n".join(gt_words), "\n".join(pred_words))
        total_words += max(1, len(gt_words))

    cer = total_char_dist / total_chars
    wer = total_word_dist / total_words

    print(f"Samples: {len(pairs)}")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")


if __name__ == "__main__":
    main()
