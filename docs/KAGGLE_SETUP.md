# Kaggle Setup (Training/Fine-tuning)

This project is organized so model training happens in Kaggle and inference can run either in Kaggle or locally.

## 1. Kaggle notebook settings

- Enable GPU in notebook settings.
- Add this repository as a Kaggle Dataset or clone from GitHub in a code cell.

## 2. Folder expectations in Kaggle

Target working path:

```text
/kaggle/working/Emerge3
```

Expected files before training:

- `data/lists/train.txt`
- `data/lists/val.txt`
- `configs/rec/te/te_char_dict.txt`

## 3. Install dependencies

```bash
cd /kaggle/working/Emerge3
bash kaggle/setup_kaggle.sh
```

Then install the correct PaddlePaddle build for the current Kaggle CUDA runtime.

## 4. Prepare data lists

If you have one master annotation file:

```bash
python scripts/prepare_rec_lists.py \
  --annotations data/processed/annotations.tsv \
  --output-dir data/lists \
  --val-ratio 0.15
```

Build Telugu char dictionary from labels:

```bash
python scripts/build_te_char_dict.py \
  --label-files data/lists/train.txt data/lists/val.txt \
  --output configs/rec/te/te_char_dict.txt
```

## 5. Run training

```bash
bash kaggle/run_train.sh
```

Model output:

- `models/te_ppocrv3/`

Train logs:

- `logs/terminal/train_*.log`

## 6. Run evaluation

```bash
bash kaggle/run_eval.sh models/te_ppocrv3/best_accuracy
```

Eval logs:

- `logs/terminal/eval_*.log`

## 7. Export model for inference (optional)

Use PaddleOCR export command directly after identifying best checkpoint.

## 8. Kaggle persistence note

Kaggle runtime storage is ephemeral. Save final model artifacts to:

- Kaggle output
- Kaggle Dataset version
- External storage (Drive/S3/etc.)
