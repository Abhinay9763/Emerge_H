#!/bin/bash
set -euo pipefail

CONFIG=configs/rec/te/te_PP-OCRv3_rec.yml
LOG_FILE=logs/terminal/train_$(date +%Y%m%d_%H%M%S).log

mkdir -p logs/terminal

echo "Starting training with ${CONFIG}"
python PaddleOCR/tools/train.py -c ${CONFIG} 2>&1 | tee ${LOG_FILE}

echo "Training complete. Log saved to ${LOG_FILE}"
