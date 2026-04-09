import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a character dictionary from PaddleOCR label files"
    )
    parser.add_argument(
        "--label-files",
        nargs="+",
        type=Path,
        required=True,
        help="One or more label files in image_path<TAB>text format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/rec/te/te_char_dict.txt"),
        help="Output character dictionary path",
    )
    return parser.parse_args()


def collect_characters(label_files):
    charset = set()
    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                _, text = line.split("\t", maxsplit=1)
                for ch in text:
                    charset.add(ch)
    return sorted(charset)


def main() -> None:
    args = parse_args()
    charset = collect_characters(args.label_files)
    if not charset:
        raise ValueError("No characters found. Check your label files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ch in charset:
            f.write(ch + "\n")

    print(f"Wrote {len(charset)} characters to {args.output}")


if __name__ == "__main__":
    main()
