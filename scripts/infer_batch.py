import argparse
import json
from pathlib import Path

from paddleocr import PaddleOCR


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Telugu OCR on an image folder")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument("--rec-model-dir", type=Path, required=True)
    parser.add_argument(
        "--rec-char-dict-path",
        type=Path,
        default=Path("configs/rec/te/te_char_dict.txt"),
    )
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def iter_images(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def main() -> None:
    args = parse_args()

    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="te",
        det=False,
        rec=True,
        use_gpu=args.use_gpu,
        rec_model_dir=str(args.rec_model_dir),
        rec_char_dict_path=str(args.rec_char_dict_path),
        show_log=False,
    )

    predictions = []
    for image_path in iter_images(args.input_dir):
        result = ocr.ocr(str(image_path), cls=False)
        lines = []
        if result and result[0]:
            for item in result[0]:
                text, score = item[0], item[1]
                lines.append({"text": text, "score": float(score)})

        predictions.append(
            {
                "image": str(image_path),
                "lines": lines,
                "joined_text": " ".join(line["text"] for line in lines),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(predictions)} predictions to {args.output_json}")


if __name__ == "__main__":
    main()
