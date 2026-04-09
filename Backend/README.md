Backend Service

Responsibilities
- Accept uploads and validate file type/size.
- Store records and status in SQLite.
- Run OCR asynchronously by calling ML inference contract.

Important Note
- No OCR model logic is implemented in Backend.
- Backend imports ML.inference.predict.predict_page at startup.

Run
1. pip install -r requirements.txt
2. Also install ML dependencies from project root:
	pip install -r ../ML/requirements.txt
3. Start server from project root:
	uvicorn Backend.main:app --reload

Health
- GET /health returns pipeline_loaded and db_connected.

Failure Mode
- If ML dependencies are missing, Backend falls back to a mock predictor and logs a warning.

