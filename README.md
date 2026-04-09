Emerge_H: Telugu Manuscript OCR MVP

This project provides a FastAPI backend for image upload and OCR archiving, with all OCR logic isolated in the ML module.

Architecture
- Backend: API, upload validation, background task orchestration, SQLite persistence.
- ML: PaddleOCR Telugu inference, preprocessing, visualization, and CER/WER benchmarking.

Project Structure
- Backend/
- ML/
- main.py

Quick Start (Local)
1. Install backend dependencies:
	pip install -r Backend/requirements.txt
2. Install ML dependencies:
	pip install -r ML/requirements.txt
3. Start API:
	uvicorn Backend.main:app --reload

API Endpoints
- POST /transcribe
- GET /archive
- GET /archive/{id}
- GET /health

Baseline Evaluation (No Fine-Tune)
1. Create JSONL manifest under ML/data/manifests/ with image_path and text.
2. Run benchmark:
	python ML/experiments/benchmark.py --manifest ML/data/manifests/your_val.jsonl
3. Check generated report under ML/evaluation/reports/.

CER Decision Gate
- CER <= 0.25: keep baseline for demo.
- 0.25 < CER <= 0.40: tune preprocessing and thresholds once.
- CER > 0.40: consider optional fine-tuning after end-to-end API demo is stable.

