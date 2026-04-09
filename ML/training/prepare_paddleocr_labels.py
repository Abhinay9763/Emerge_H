from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolve_image(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    resolved = (manifest_path.parent / candidate).resolve()
    if resolved.exists():
        return resolved

    parts = list(candidate.parts)
    if "TeluguSeg" in parts:
        idx = parts.index("TeluguSeg")
        nested_parts = parts[: idx + 1] + ["TeluguSeg"] + parts[idx + 1 :]
        nested = (manifest_path.parent / Path(*nested_parts)).resolve()
        if nested.exists():
            return nested

    return resolved


def _convert_manifest(manifest_path: Path, data_root: Path, out_txt: Path) -> int:
    count = 0
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as in_f, out_txt.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            image_path = _resolve_image(manifest_path, str(row["image_path"]))
            text = str(row.get("text", "")).replace("\t", " ").replace("\n", " ").strip()
            if not text:
                continue
            if not image_path.exists():
                continue

            try:
                rel_path = image_path.resolve().relative_to(data_root.resolve())
            except ValueError:
                # Skip files outside the configured data root.
                continue

            out_f.write(f"{rel_path.as_posix()}\t{text}\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL manifest to PaddleOCR rec label txt")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    train_manifest = Path(args.train_manifest).resolve()
    val_manifest = Path(args.val_manifest).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    train_out = out_dir / "rec_gt_train.txt"
    val_out = out_dir / "rec_gt_val.txt"

    train_n = _convert_manifest(train_manifest, data_root, train_out)
    val_n = _convert_manifest(val_manifest, data_root, val_out)

    print(f"Wrote {train_out} ({train_n} rows)")
    print(f"Wrote {val_out} ({val_n} rows)")


if __name__ == "__main__":
    main()
