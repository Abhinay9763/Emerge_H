import sqlite3
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "transcriptions.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def init_db() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                ocr_image_path TEXT,
                telugu TEXT NOT NULL,
                english TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(transcriptions)").fetchall()}
        if "status" not in columns:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN status TEXT NOT NULL DEFAULT 'done'")
        if "ocr_image_path" not in columns:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN ocr_image_path TEXT")

        conn.commit()


def insert(image_path: str, telugu: str, english: str, status: str) -> int:
    created_at = _utc_iso_now()
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO transcriptions (image_path, ocr_image_path, telugu, english, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_path, None, telugu, english, status, created_at),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_all(limit: int, offset: int) -> tuple[int, list[dict]]:
    with _get_connection() as conn:
        total_row = conn.execute("SELECT COUNT(*) AS total FROM transcriptions").fetchone()
        rows = conn.execute(
            """
            SELECT id, image_path, ocr_image_path, telugu, english, status, created_at
            FROM transcriptions
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    total = int(total_row["total"]) if total_row else 0
    items = [dict(row) for row in rows]
    return total, items


def get_by_id(id: int) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, image_path, ocr_image_path, telugu, english, status, created_at
            FROM transcriptions
            WHERE id = ?
            """,
            (id,),
        ).fetchone()

    return dict(row) if row else None
