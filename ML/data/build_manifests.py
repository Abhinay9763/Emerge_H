from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "IIIT-HW-Telugu_v1"
MANIFEST_DIR = ROOT / "manifests"

SPLITS = {
    "train": DATASET_DIR / "train.txt",
    "val": DATASET_DIR / "val.txt",
    "test": DATASET_DIR / "test.txt",
}


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None

    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None

    return parts[0], parts[1]


def build_manifest(split: str, source_file: Path) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MANIFEST_DIR / f"cvit_{split}.jsonl"

    records_written = 0
    with source_file.open("r", encoding="utf-8") as in_f, output_path.open("w", encoding="utf-8") as out_f:
        for raw in in_f:
            parsed = _parse_line(raw)
            if parsed is None:
                continue

            image_rel, label = parsed
            record = {
                "image_path": f"../IIIT-HW-Telugu_v1/{image_rel}",
                "text": label,
                "split": split,
                "source": "cvit",
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    print(f"Built {output_path} with {records_written} rows")
    return output_path


def main() -> None:
    for split, src in SPLITS.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing split file: {src}")
        build_manifest(split, src)


if __name__ == "__main__":
    main()
