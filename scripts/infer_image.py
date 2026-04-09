import argparse
from pathlib import Path

from paddleocr import PaddleOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Telugu OCR on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--rec-model-dir",
        type=Path,
        required=True,
        help="Path to trained recognition model inference directory",
    )
    parser.add_argument(
        "--rec-char-dict-path",
        type=Path,
        default=Path("configs/rec/te/te_char_dict.txt"),
    )
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


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

    result = ocr.ocr(str(args.image), cls=False)

    if not result or not result[0]:
        print("No text recognized")
        return

    for idx, item in enumerate(result[0], start=1):
        text, score = item[0], item[1]
        print(f"{idx}. {text} (conf={score:.4f})")


if __name__ == "__main__":
    main()
