# Terminal Logs Guide

Use this guide to keep clean, judge-friendly execution logs.

## Why logs matter

- Reproducibility
- Debug traceability
- Evidence for hackathon judges

## Training and eval logs

These scripts already log output to timestamped files:

- `kaggle/run_train.sh`
- `kaggle/run_eval.sh`

Log location:

- `logs/terminal/`

## Capturing setup logs

Use `tee` while running setup:

```bash
bash kaggle/setup_kaggle.sh 2>&1 | tee logs/terminal/setup_$(date +%Y%m%d_%H%M%S).log
```

## Capturing dataset prep logs

```bash
python scripts/prepare_rec_lists.py \
  --annotations data/processed/annotations.tsv \
  --output-dir data/lists \
  --val-ratio 0.15 2>&1 | tee logs/terminal/data_prep_$(date +%Y%m%d_%H%M%S).log
```

## Suggested log naming convention

- `setup_YYYYMMDD_HHMMSS.log`
- `data_prep_YYYYMMDD_HHMMSS.log`
- `train_YYYYMMDD_HHMMSS.log`
- `eval_YYYYMMDD_HHMMSS.log`
- `infer_YYYYMMDD_HHMMSS.log`

## What to include in final submission

- One setup log
- One successful training log
- One evaluation log showing key metric
- One inference log on demo images
