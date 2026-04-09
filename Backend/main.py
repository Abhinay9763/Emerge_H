import asyncio
import logging
import sqlite3
from io import BytesIO
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    from . import database
except ImportError:
    import database


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
ML_TIMEOUT_SECONDS = 30


logger = logging.getLogger("anciflow")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _mock_predict_page(_img: Image.Image) -> dict[str, str]:
    return {"telugu": "మాక్ తెలుగు వచనం", "english": "mock Telugu text"}


predict_page: Callable[[Image.Image], dict[str, str]] = _mock_predict_page


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predict_page

    database.init_db()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ML.inference.predict import predict_page as pipeline_predict_page

        predict_page = pipeline_predict_page
        app.state.pipeline_loaded = True
    except Exception as exc:
        logger.warning("ML pipeline not loaded, using mock predictor: %s", exc)
        predict_page = _mock_predict_page
        app.state.pipeline_loaded = False

    yield


app = FastAPI(title="AnciFlow Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(UPLOADS_DIR), check_dir=False), name="images")


def _to_response_item(row: dict) -> dict:
    image_path = str(row["image_path"]).replace("\\", "/")
    ocr_image_path = row.get("ocr_image_path")
    return {
        "id": row["id"],
        "telugu": row["telugu"],
        "english": row["english"],
        "status": row["status"],
        "image_url": f"/images/{image_path}",
        "ocr_image_url": f"/images/{str(ocr_image_path).replace('\\', '/')}" if ocr_image_path else None,
        "created_at": row["created_at"],
    }


def _build_upload_paths(content_type: str) -> tuple[Path, str]:
    now = datetime.now(timezone.utc)
    subdir = Path(now.strftime("%Y")) / now.strftime("%m")
    ext = ALLOWED_CONTENT_TYPES[content_type]
    filename = f"{uuid4().hex}{ext}"
    relative_path = (subdir / filename).as_posix()
    absolute_path = UPLOADS_DIR / subdir / filename
    return absolute_path, relative_path


async def _save_upload_file(file: UploadFile, destination: Path) -> None:
    total_size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(destination, "wb") as out_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                await out_file.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=422, detail="File too large. Max size is 10MB.")

            await out_file.write(chunk)


async def _verify_image_file(path: Path) -> None:
    def _verify() -> None:
        with Image.open(path) as img:
            img.verify()
            if img.format not in {"JPEG", "PNG"}:
                raise ValueError("Unsupported image format")

    try:
        await run_in_threadpool(_verify)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid or corrupted image.")


def _update_transcription(
    record_id: int,
    telugu: str,
    english: str,
    status: str,
    ocr_image_path: str | None = None,
) -> None:
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.execute(
            """
            UPDATE transcriptions
            SET telugu = ?, english = ?, status = ?, ocr_image_path = ?
            WHERE id = ?
            """,
            (telugu, english, status, ocr_image_path, record_id),
        )
        conn.commit()


def _db_connected() -> bool:
    try:
        with sqlite3.connect(database.DB_PATH) as conn:
            conn.execute("SELECT 1")
        return True
    except sqlite3.Error:
        return False


async def _run_pipeline(path: Path) -> dict[str, str]:
    def _predict() -> dict[str, str]:
        with Image.open(path) as img:
            return predict_page(img)

    return await asyncio.wait_for(run_in_threadpool(_predict), timeout=ML_TIMEOUT_SECONDS)


def _save_ocr_image_asset(payload: object) -> str | None:
    if payload is None:
        return None

    now = datetime.now(timezone.utc)
    subdir = Path(now.strftime("%Y")) / now.strftime("%m")
    filename = f"ocr_{uuid4().hex}.png"
    rel_path = (subdir / filename).as_posix()
    abs_path = UPLOADS_DIR / subdir / filename
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, Image.Image):
        payload.save(abs_path, format="PNG")
        return rel_path

    if isinstance(payload, (bytes, bytearray)):
        with Image.open(BytesIO(payload)) as img:
            img.save(abs_path, format="PNG")
        return rel_path

    if isinstance(payload, str):
        candidate = Path(payload)
        if candidate.is_file():
            with Image.open(candidate) as img:
                img.save(abs_path, format="PNG")
            return rel_path

    return None


async def _process_transcription(record_id: int, image_path: Path) -> None:
    logger.info("ML start: record_id=%s image=%s", record_id, image_path)

    try:
        prediction = await _run_pipeline(image_path)
        telugu = str(prediction.get("telugu", ""))
        english = str(prediction.get("english", ""))
        ocr_payload = prediction.get("ocr_image") or prediction.get("ocr_image_bytes") or prediction.get("ocr_image_path")
        ocr_image_path = await run_in_threadpool(_save_ocr_image_asset, ocr_payload)
        _update_transcription(record_id, telugu, english, "done", ocr_image_path=ocr_image_path)
        logger.info("ML end: record_id=%s status=done", record_id)
    except Exception as exc:
        logger.error("ML failed: record_id=%s error=%s", record_id, exc)
        _update_transcription(record_id, "", "", "failed")


@app.get("/")
def root():
    return {
        "name": "AnciFlow API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/transcribe")
async def transcribe(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported file type. Send JPEG or PNG.")

    save_path, relative_path = _build_upload_paths(file.content_type)

    try:
        await _save_upload_file(file, save_path)
        await _verify_image_file(save_path)
        logger.info("Upload saved: path=%s", save_path)
    finally:
        await file.close()

    try:
        record_id = database.insert(relative_path, "", "", "processing")
        background_tasks.add_task(_process_transcription, record_id, save_path)
    except Exception:
        save_path.unlink(missing_ok=True)
        logger.exception("Failed to queue transcription")
        raise HTTPException(status_code=500, detail="Pipeline failed: Unable to queue processing")

    row = database.get_by_id(record_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Pipeline failed: Record insert verification failed")

    total, _ = database.get_all(limit=1, offset=0)
    return {
        "total": total,
        "items": [_to_response_item(row)],
    }


@app.get("/archive")
def archive(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    total, rows = database.get_all(limit=limit, offset=offset)
    return {"total": total, "items": [_to_response_item(row) for row in rows]}


@app.get("/archive/{id}")
def archive_item(id: int):
    row = database.get_by_id(id)
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found.")
    return _to_response_item(row)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": bool(getattr(app.state, "pipeline_loaded", False)),
        "db_connected": _db_connected(),
    }
