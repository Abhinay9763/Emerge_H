ML Module (PaddleOCR Telugu)

Scope
- All OCR and evaluation logic lives here.
- Backend should only call predict_page via ML interface.

Contents
- inference/preprocess.py: lightweight manuscript normalization.
- inference/visualize.py: OCR box overlay rendering.
- inference/predict.py: PaddleOCR Telugu inference contract.
- evaluation/metrics.py: CER and WER utilities.
- experiments/benchmark.py: baseline evaluation runner.

Install
pip install -r ML/requirements.txt

Manifest Format
Create JSONL files under ML/data/manifests/.
Required fields per line:
- image_path
- text

Generate manifests from CVIT split files
python ML/data/build_manifests.py

Example
{"image_path": "../../datasets/cvit/val/sample1.png", "text": "తెలుగు గ్రౌండ్ ట్రూత్"}

Run Baseline Benchmark
python ML/experiments/benchmark.py --manifest ML/data/manifests/cvit_val.jsonl

Output
- JSON report written to ML/evaluation/reports/ with per-sample CER/WER and summary latency.

