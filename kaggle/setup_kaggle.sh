#!/bin/bash
set -euo pipefail

echo "[1/5] Updating pip"
pip install --upgrade pip

echo "[2/5] Installing PaddlePaddle (choose version compatible with Kaggle CUDA)"
# Example (CUDA 12.x):
# pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# CPU fallback:
# pip install paddlepaddle==2.6.1

echo "[3/5] Cloning PaddleOCR"
if [ ! -d "PaddleOCR" ]; then
  git clone https://github.com/PaddlePaddle/PaddleOCR.git
fi

echo "[4/5] Installing PaddleOCR requirements"
pip install -r PaddleOCR/requirements.txt

echo "[5/5] Installing project requirements"
pip install -r kaggle/requirements_kaggle.txt

echo "Kaggle setup complete"
