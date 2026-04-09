#!/bin/bash
set -euo pipefail

CONFIG=configs/rec/te/te_PP-OCRv3_rec.yml
CHECKPOINT=${1:-models/te_ppocrv3/latest}
LOG_FILE=logs/terminal/eval_$(date +%Y%m%d_%H%M%S).log

mkdir -p logs/terminal

echo "Evaluating checkpoint: ${CHECKPOINT}"
python PaddleOCR/tools/eval.py -c ${CONFIG} -o Global.checkpoints=${CHECKPOINT} 2>&1 | tee ${LOG_FILE}

echo "Evaluation complete. Log saved to ${LOG_FILE}"
