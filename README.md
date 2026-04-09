# Telugu Handwritten OCR (PaddleOCR)

Hackathon-friendly OCR pipeline for Telugu handwritten manuscript digitization.

This repository is organized for:
- Fast fine-tuning in Kaggle
- Clean separation between training and inference
- Reproducible logs and documented workflow

## Project structure

```text
Emerge3/
  configs/
    rec/te/
      te_PP-OCRv3_rec.yml
      te_char_dict.txt
  data/
    raw/
    processed/
    lists/
  docs/
    WALKTHROUGH.md
    KAGGLE_SETUP.md
    TERMINAL_LOGS.md
  kaggle/
    setup_kaggle.sh
    run_train.sh
    run_eval.sh
    requirements_kaggle.txt
  logs/
    terminal/
      kaggle_run_log_template.md
  models/
  outputs/
  scripts/
    prepare_rec_lists.py
    build_te_char_dict.py
    infer_image.py
    infer_batch.py
    eval_cer.py
  .gitignore
  requirements.txt
```

## Quickstart

1. Read `docs/KAGGLE_SETUP.md`.
2. Upload this repo to Kaggle as a dataset or clone from GitHub.
3. Follow commands in `kaggle/setup_kaggle.sh` and `kaggle/run_train.sh`.
4. Use `scripts/infer_image.py` or `scripts/infer_batch.py` for inference.

## Data format (recognition)

Each line in train/val txt files:

```text
/path/to/image.jpg<TAB>ground_truth_text
```

## Notes

- Training is designed to run in Kaggle.
- Local machine can be used for inference and quick sanity checks.
- Keep `configs/rec/te/te_char_dict.txt` aligned with your label vocabulary.
