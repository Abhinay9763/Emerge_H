Telugu Fine-Tuning (PaddleOCR Guidelines)

This folder contains a repo-based fine-tuning flow for CVIT Telugu data.

Files
- prepare_paddleocr_labels.py: converts project JSONL manifests into PaddleOCR rec label txt files.
- run_finetune_te.ps1: clones PaddleOCR and launches fine-tuning with Telugu config.

Run
1. From project root:
   .\\.venv312\\Scripts\\Activate.ps1
2. From project root:
   powershell -ExecutionPolicy Bypass -File .\\ML\\training\\run_finetune_te.ps1

Optional arguments
- Epochs:
  powershell -ExecutionPolicy Bypass -File .\\ML\\training\\run_finetune_te.ps1 -Epochs 30
- Batch size:
  powershell -ExecutionPolicy Bypass -File .\\ML\\training\\run_finetune_te.ps1 -BatchSize 64
- Start from a pretrained checkpoint:
  powershell -ExecutionPolicy Bypass -File .\\ML\\training\\run_finetune_te.ps1 -PretrainedModel "C:/path/to/checkpoint"

Notes
- The script uses config: PaddleOCR/configs/rec/PP-OCRv3/multi_language/te_PP-OCRv3_mobile_rec.yml
- Labels are generated into ML/training/labels.
- Model outputs are written into ML/training/artifacts/te_finetune.
- Run logs are written into ML/training/artifacts/logs:
  - finetune_YYYYMMDD_HHMMSS.log (high-level step log)
  - train_YYYYMMDD_HHMMSS.log (full training stdout/stderr)

Monitor logs live (PowerShell)
Get-Content .\\ML\\training\\artifacts\\logs\\train_*.log -Wait
