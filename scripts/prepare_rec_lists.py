import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PaddleOCR train/val list files from TSV annotations"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="TSV file with columns: image_path<TAB>text",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lists"),
        help="Output directory for train.txt and val.txt",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            img_path, text = parts
            rows.append((img_path, text))
    return rows


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for img_path, text in rows:
            f.write(f"{img_path}\t{text}\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.annotations)
    if not rows:
        raise ValueError("No valid rows found in annotations file")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val_size = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.txt"
    val_path = args.output_dir / "val.txt"

    write_rows(train_path, train_rows)
    write_rows(val_path, val_rows)

    print(f"Total rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)} -> {train_path}")
    print(f"Val rows:   {len(val_rows)} -> {val_path}")


if __name__ == "__main__":
    main()
