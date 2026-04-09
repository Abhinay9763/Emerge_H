# End-to-End Walkthrough (Telugu + PaddleOCR)

This file explains exactly what each part of the pipeline does.

## Goal

Fine-tune a PaddleOCR recognition model for Telugu handwritten manuscript text in Kaggle.

## Pipeline summary

1. Collect line-level images and text labels.
2. Build train/val split files.
3. Build Telugu character dictionary from labels.
4. Fine-tune recognition model with PaddleOCR config.
5. Evaluate model on validation split.
6. Run inference for one image or a folder.
7. Track all commands and outcomes in terminal logs.

## Step-by-step details

### Step 1: Dataset preparation

Input requirement:

- Line image files
- An annotation file in this format:

```text
/path/to/line_001.jpg<TAB>ground_truth_text
```

File used:

- `scripts/prepare_rec_lists.py`

What it does:

- Reads a TSV annotations file
- Shuffles rows reproducibly (seed)
- Splits into train and val
- Writes:
  - `data/lists/train.txt`
  - `data/lists/val.txt`

### Step 2: Character dictionary generation

File used:

- `scripts/build_te_char_dict.py`

What it does:

- Parses all label text from train/val files
- Extracts unique characters
- Writes sorted dictionary to:
  - `configs/rec/te/te_char_dict.txt`

Why this matters:

- The model output classes must align exactly with your Telugu character set.

### Step 3: Training config

File used:

- `configs/rec/te/te_PP-OCRv3_rec.yml`

What it defines:

- Model architecture (SVTR_LCNet recognition)
- Input resize settings
- Train/eval file paths
- Optimizer and learning rate schedule
- Save and eval intervals

### Step 4: Kaggle environment setup

File used:

- `kaggle/setup_kaggle.sh`

What it does:

- Upgrades pip
- Clones PaddleOCR
- Installs PaddleOCR requirements
- Installs project requirements
- Leaves PaddlePaddle install line for CUDA-compatible selection

### Step 5: Train

File used:

- `kaggle/run_train.sh`

What it does:

- Runs `PaddleOCR/tools/train.py` with Telugu config
- Saves stdout/stderr to timestamped log file under `logs/terminal/`

### Step 6: Evaluate

File used:

- `kaggle/run_eval.sh`

What it does:

- Evaluates a specified checkpoint using same config
- Writes eval logs to `logs/terminal/`

### Step 7: Inference

Files used:

- `scripts/infer_image.py`
- `scripts/infer_batch.py`

What they do:

- Load your trained recognition model
- Run OCR on one image or folder
- Print text + confidence
- Save batch JSON predictions

### Step 8: Error metrics

File used:

- `scripts/eval_cer.py`

What it does:

- Computes CER and WER from a TSV file (`gt<TAB>pred`)
- Useful for benchmark reporting in your hackathon demo

## Minimal command flow

```bash
cd /kaggle/working/Emerge3
bash kaggle/setup_kaggle.sh

python scripts/prepare_rec_lists.py \
  --annotations data/processed/annotations.tsv \
  --output-dir data/lists \
  --val-ratio 0.15

python scripts/build_te_char_dict.py \
  --label-files data/lists/train.txt data/lists/val.txt \
  --output configs/rec/te/te_char_dict.txt

bash kaggle/run_train.sh
bash kaggle/run_eval.sh models/te_ppocrv3/best_accuracy
```

## Inference command examples

Single image:

```bash
python scripts/infer_image.py \
  --image data/raw/sample.jpg \
  --rec-model-dir models/te_ppocrv3/inference \
  --rec-char-dict-path configs/rec/te/te_char_dict.txt \
  --use-gpu
```

Batch folder:

```bash
python scripts/infer_batch.py \
  --input-dir data/raw \
  --output-json outputs/predictions.json \
  --rec-model-dir models/te_ppocrv3/inference \
  --rec-char-dict-path configs/rec/te/te_char_dict.txt \
  --use-gpu
```

## What to show in hackathon demo

- Baseline vs fine-tuned sample outputs
- CER/WER improvement
- Low-confidence samples for transparency
- Log snippet proving reproducible training run
