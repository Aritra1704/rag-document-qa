"""PostgreSQL storage and OCR-structure parsing helpers for PDF Name Search."""

from __future__ import annotations

import os
import re
import io
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence
from urllib.parse import quote_plus
try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except Exception as import_exc:  # noqa: BLE001
    psycopg = None
    sql = None
    dict_row = None
    _PSYCOPG_IMPORT_ERROR = import_exc
else:
    _PSYCOPG_IMPORT_ERROR = None


SCHEMA_SQL_PATH = Path(__file__).resolve().parent.parent / "db" / "postgres" / "001_name_search_schema.sql"
INDEX_SQL_PATH = Path(__file__).resolve().parent.parent / "db" / "postgres" / "002_name_search_indexes.sql"
PROJECT_SCHEMA = "rag_document_qa"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_ENV_LOADED = False
_CV2_MODULE = None
_NP_MODULE = None
_CV2_IMPORT_ERROR: str | None = None


def _load_cv2_numpy():
    global _CV2_MODULE
    global _NP_MODULE
    global _CV2_IMPORT_ERROR
    if _CV2_MODULE is not None and _NP_MODULE is not None:
        return _CV2_MODULE, _NP_MODULE, None
    if _CV2_IMPORT_ERROR is not None:
        return None, None, _CV2_IMPORT_ERROR

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _CV2_IMPORT_ERROR = str(exc)
        return None, None, _CV2_IMPORT_ERROR

    _CV2_MODULE = cv2
    _NP_MODULE = np
    return _CV2_MODULE, _NP_MODULE, None


def _load_local_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    load_dotenv(dotenv_path=ENV_PATH, override=False)
    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    _ENV_LOADED = True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve PostgreSQL connection URL from explicit input or environment variables."""
    _load_local_env()

    candidate = (database_url or "").strip()
    if candidate:
        return _normalize_database_url(candidate)

    env_database_url = os.environ.get("DATABASE_URL", "").strip()
    if env_database_url:
        return _normalize_database_url(env_database_url)

    pg_host = os.environ.get("PGHOST", "").strip() or "localhost"
    pg_port = os.environ.get("PGPORT", "").strip() or "5432"
    pg_database = os.environ.get("PGDATABASE", "").strip()
    pg_user = os.environ.get("PGUSER", "").strip()
    pg_password = os.environ.get("PGPASSWORD", "")

    if not pg_database or not pg_user:
        raise ValueError(
            "PostgreSQL configuration is missing. Set DATABASE_URL, "
            "or set PGDATABASE and PGUSER (with optional PGHOST/PGPORT/PGPASSWORD)."
        )

    encoded_user = quote_plus(pg_user)
    if pg_password:
        encoded_password = quote_plus(pg_password)
        auth_segment = f"{encoded_user}:{encoded_password}"
    else:
        auth_segment = encoded_user

    return f"postgresql://{auth_segment}@{pg_host}:{pg_port}/{pg_database}"


def _normalize_database_url(database_url: str) -> str:
    stripped = database_url.strip()
    if stripped.startswith("postgres://"):
        return "postgresql://" + stripped[len("postgres://") :]
    return stripped


def _load_sql_script(script_path: Path) -> str:
    if not script_path.exists():
        raise RuntimeError(f"Missing PostgreSQL schema script: {script_path}")
    return script_path.read_text(encoding="utf-8")


def _ensure_project_schema(connection) -> None:
    if sql is None:
        return
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(PROJECT_SCHEMA)))
        cursor.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(PROJECT_SCHEMA)))
    connection.commit()


def _execute_sql_script(connection, script_text: str) -> None:
    statements: list[str] = []
    statement_lines: list[str] = []
    for raw_line in script_text.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("--"):
            continue
        statement_lines.append(raw_line)
        if stripped_line.endswith(";"):
            statement_sql = "\n".join(statement_lines).strip()
            if statement_sql:
                statements.append(statement_sql)
            statement_lines = []

    if statement_lines:
        trailing_sql = "\n".join(statement_lines).strip()
        if trailing_sql:
            statements.append(trailing_sql)

    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)


def open_storage_connection(database_url: str | None = None):
    """Open PostgreSQL connection and ensure required tables/indexes exist."""

    if psycopg is None:
        raise RuntimeError(f"PostgreSQL driver unavailable: {_PSYCOPG_IMPORT_ERROR}")

    resolved_database_url = resolve_database_url(database_url)
    connection = psycopg.connect(resolved_database_url, row_factory=dict_row)
    _ensure_project_schema(connection)
    initialize_storage_schema(connection)
    _ensure_project_schema(connection)
    return connection


def initialize_storage_schema(connection) -> None:
    schema_sql = _load_sql_script(SCHEMA_SQL_PATH)
    index_sql = _load_sql_script(INDEX_SQL_PATH)
    _execute_sql_script(connection, schema_sql)
    _execute_sql_script(connection, index_sql)
    connection.commit()


def upsert_document(
    connection,
    *,
    folder_path: str,
    file_name: str,
    file_path: str,
    pages_total: int,
    status: str,
    error_message: str | None = None,
) -> int:
    try:
        file_stat = Path(file_path).stat()
        last_modified = float(file_stat.st_mtime)
        file_size = int(file_stat.st_size)
    except OSError:
        last_modified = None
        file_size = None

    now = _utc_now_iso()
    row = connection.execute(
        """
        INSERT INTO documents (
            folder_path, file_name, file_path, pages_total, pages_processed,
            status, error_message, last_modified, file_size,
            last_processed_at, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, 0, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT(file_path) DO UPDATE SET
            folder_path = EXCLUDED.folder_path,
            file_name = EXCLUDED.file_name,
            pages_total = EXCLUDED.pages_total,
            status = EXCLUDED.status,
            error_message = EXCLUDED.error_message,
            last_modified = EXCLUDED.last_modified,
            file_size = EXCLUDED.file_size,
            last_processed_at = EXCLUDED.last_processed_at,
            updated_at = NOW()
        RETURNING id
        """,
        (
            folder_path,
            file_name,
            file_path,
            int(pages_total),
            status,
            error_message,
            last_modified,
            file_size,
            now,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert document: {file_path}")
    connection.commit()
    return int(row["id"])


def update_document_status(
    connection,
    *,
    document_id: int,
    status: str,
    pages_processed: int | None = None,
    error_message: str | None = None,
) -> None:
    now = _utc_now_iso()
    if pages_processed is None:
        connection.execute(
            """
            UPDATE documents
            SET status = %s, error_message = %s, last_processed_at = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (status, error_message, now, int(document_id)),
        )
    else:
        connection.execute(
            """
            UPDATE documents
            SET status = %s, pages_processed = %s, error_message = %s,
                last_processed_at = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (status, int(pages_processed), error_message, now, int(document_id)),
        )
    connection.commit()


def upsert_page(
    connection,
    *,
    document_id: int,
    page_number: int,
    status: str,
    extraction_method: str | None,
    raw_text: str,
    parsed_record_count: int,
    error_message: str | None = None,
) -> int:
    now = _utc_now_iso()
    row = connection.execute(
        """
        INSERT INTO pages (
            document_id, page_number, status, extraction_method, raw_text,
            parsed_record_count, error_message, processed_at, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT(document_id, page_number) DO UPDATE SET
            status = EXCLUDED.status,
            extraction_method = EXCLUDED.extraction_method,
            raw_text = EXCLUDED.raw_text,
            parsed_record_count = EXCLUDED.parsed_record_count,
            error_message = EXCLUDED.error_message,
            processed_at = EXCLUDED.processed_at,
            updated_at = NOW()
        RETURNING id
        """,
        (
            int(document_id),
            int(page_number),
            status,
            extraction_method,
            raw_text,
            int(parsed_record_count),
            error_message,
            now,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert page {page_number} for document {document_id}")
    connection.commit()
    return int(row["id"])


def replace_page_records(
    connection,
    *,
    document_id: int,
    page_id: int,
    records: Sequence[dict[str, Any]],
    page_number: int | None = None,
    extraction_method: str | None = None,
) -> None:
    connection.execute("DELETE FROM parsed_records WHERE page_id = %s", (int(page_id),))
    insert_page_records(
        connection,
        document_id=document_id,
        page_id=page_id,
        records=records,
        page_number=page_number,
        extraction_method=extraction_method,
    )
    connection.commit()


def insert_page_records(
    connection,
    *,
    document_id: int,
    page_id: int,
    records: Sequence[dict[str, Any]],
    page_number: int | None = None,
    extraction_method: str | None = None,
) -> None:
    now = _utc_now_iso()
    forced_page_number = int(page_number) if page_number is not None else None
    forced_extraction_method = (extraction_method or "").strip()
    for record in records:
        name_value = str(record.get("name") or "")
        record_page_number = forced_page_number
        if record_page_number is None:
            raw_page_number = record.get("page_number")
            if raw_page_number is None:
                raise ValueError("Parsed record missing page_number for parsed_records insert.")
            record_page_number = int(raw_page_number)
        if int(record_page_number) <= 0:
            raise ValueError(f"Invalid parsed record page_number: {record_page_number}")

        record_extraction_method = forced_extraction_method or str(record.get("extraction_method") or "").strip()
        if not record_extraction_method:
            record_extraction_method = "unknown"

        connection.execute(
            """
            INSERT INTO parsed_records (
                document_id, page_id,
                serial_number, elector_id, name, relative_name, relative_type,
                house_number, age, gender, constituency, section_name,
                file_name, file_path, page_number, extraction_method,
                raw_record_text, name_normalized, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """,
            (
                int(document_id),
                int(page_id),
                record.get("serial_number"),
                record.get("elector_id"),
                name_value or None,
                record.get("relative_name"),
                record.get("relative_type"),
                record.get("house_number"),
                record.get("age"),
                record.get("gender"),
                record.get("constituency"),
                record.get("section_name"),
                record.get("file_name") or "",
                record.get("file_path") or "",
                int(record_page_number),
                record_extraction_method,
                record.get("raw_record_text") or "",
                name_value.strip().lower() if name_value else None,
                now,
            ),
        )
    connection.commit()


def get_folder_storage_summary(connection, folder_path: str) -> dict[str, int]:
    docs_row = connection.execute(
        "SELECT COUNT(*) AS c FROM documents WHERE folder_path = %s",
        (folder_path,),
    ).fetchone()
    pages_row = connection.execute(
        """
        SELECT COUNT(*) AS c
        FROM pages p
        JOIN documents d ON d.id = p.document_id
        WHERE d.folder_path = %s
        """,
        (folder_path,),
    ).fetchone()
    records_row = connection.execute(
        """
        SELECT COUNT(*) AS c
        FROM parsed_records r
        JOIN documents d ON d.id = r.document_id
        WHERE d.folder_path = %s
        """,
        (folder_path,),
    ).fetchone()
    return {
        "documents": int(docs_row["c"]) if docs_row else 0,
        "pages": int(pages_row["c"]) if pages_row else 0,
        "records": int(records_row["c"]) if records_row else 0,
    }


def search_stored_records(
    connection,
    *,
    folder_path: str,
    name_query: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    normalized_query = (name_query or "").strip().lower()
    if not normalized_query:
        return []

    rows = connection.execute(
        """
        SELECT
            r.id,
            r.serial_number,
            r.elector_id,
            r.name,
            r.relative_name,
            r.relative_type,
            r.house_number,
            r.age,
            r.gender,
            r.constituency,
            r.section_name,
            r.file_name,
            r.file_path,
            r.page_number,
            r.extraction_method,
            r.raw_record_text
        FROM parsed_records r
        JOIN documents d ON d.id = r.document_id
        WHERE d.folder_path = %s
          AND (
              COALESCE(r.name_normalized, '') LIKE %s
              OR COALESCE(r.raw_record_text, '') ILIKE %s
          )
        ORDER BY r.file_path, r.page_number, r.id
        LIMIT %s
        """,
        (folder_path, f"%{normalized_query}%", f"%{normalized_query}%", int(limit)),
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_page_records_for_verification(
    connection,
    *,
    file_path: str,
    page_number: int,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            d.id AS document_id,
            p.id AS page_id,
            r.id AS parsed_record_id,
            r.serial_number,
            r.elector_id,
            r.name,
            r.relative_name,
            r.relative_type,
            r.house_number,
            r.age,
            r.gender,
            r.constituency,
            r.section_name,
            r.file_name,
            r.file_path,
            r.page_number,
            r.extraction_method,
            r.raw_record_text
        FROM parsed_records r
        JOIN pages p ON p.id = r.page_id
        JOIN documents d ON d.id = r.document_id
        WHERE d.file_path = %s
          AND p.page_number = %s
        ORDER BY r.id
        LIMIT %s
        """,
        (file_path, int(page_number), int(limit)),
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_test_page_storage_verification(
    connection,
    *,
    file_path: str,
    page_number: int,
    limit: int = 200,
) -> dict[str, Any]:
    counts_row = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM documents d WHERE d.file_path = %s) AS documents_count,
            (
                SELECT COUNT(*)
                FROM pages p
                JOIN documents d ON d.id = p.document_id
                WHERE d.file_path = %s
                  AND p.page_number = %s
            ) AS pages_count,
            (
                SELECT COUNT(*)
                FROM parsed_records r
                JOIN pages p ON p.id = r.page_id
                JOIN documents d ON d.id = r.document_id
                WHERE d.file_path = %s
                  AND p.page_number = %s
            ) AS parsed_records_count
        """,
        (
            file_path,
            file_path,
            int(page_number),
            file_path,
            int(page_number),
        ),
    ).fetchone()

    document_rows = connection.execute(
        """
        SELECT
            d.id,
            d.folder_path,
            d.file_name,
            d.file_path,
            d.pages_total,
            d.pages_processed,
            d.status,
            d.error_message,
            d.last_modified,
            d.file_size,
            d.last_processed_at,
            d.created_at,
            d.updated_at
        FROM documents d
        WHERE d.file_path = %s
        ORDER BY d.id
        LIMIT 1
        """,
        (file_path,),
    ).fetchall()

    page_rows = connection.execute(
        """
        SELECT
            p.id,
            p.document_id,
            p.page_number,
            p.status,
            p.extraction_method,
            p.raw_text,
            p.parsed_record_count,
            p.error_message,
            p.processed_at,
            p.created_at,
            p.updated_at
        FROM pages p
        JOIN documents d ON d.id = p.document_id
        WHERE d.file_path = %s
          AND p.page_number = %s
        ORDER BY p.id
        LIMIT 1
        """,
        (file_path, int(page_number)),
    ).fetchall()

    parsed_record_rows = connection.execute(
        """
        SELECT
            r.id,
            r.document_id,
            r.page_id,
            r.serial_number,
            r.elector_id,
            r.name,
            r.relative_name,
            r.relative_type,
            r.house_number,
            r.age,
            r.gender,
            r.constituency,
            r.section_name,
            r.file_name,
            r.file_path,
            r.page_number,
            r.extraction_method,
            r.raw_record_text,
            r.name_normalized,
            r.created_at,
            r.updated_at
        FROM parsed_records r
        JOIN pages p ON p.id = r.page_id
        JOIN documents d ON d.id = r.document_id
        WHERE d.file_path = %s
          AND p.page_number = %s
        ORDER BY r.id
        LIMIT %s
        """,
        (file_path, int(page_number), int(limit)),
    ).fetchall()

    documents_count = int(counts_row["documents_count"]) if counts_row else 0
    pages_count = int(counts_row["pages_count"]) if counts_row else 0
    parsed_records_count = int(counts_row["parsed_records_count"]) if counts_row else 0

    return {
        "counts": {
            "documents": documents_count,
            "pages": pages_count,
            "parsed_records": parsed_records_count,
        },
        "checks": {
            "single_document_row": documents_count == 1,
            "single_page_row": pages_count == 1,
            "multiple_parsed_records": parsed_records_count > 1,
        },
        "documents": [dict(row) for row in document_rows],
        "pages": [dict(row) for row in page_rows],
        "parsed_records": [dict(row) for row in parsed_record_rows],
    }


def _extract_context(text: str) -> tuple[str, str]:
    constituency = ""
    section_name = ""

    constituency_patterns = [
        r"(?im)\b(?:assembly\s+constituency|constituency|ac\s*name)\b\s*[:\-]?\s*([^\n]+)",
        r"(?im)\b(?:constituency\s+name)\b\s*[:\-]?\s*([^\n]+)",
    ]
    section_patterns = [
        r"(?im)\b(?:section\s*name|part\s*name|polling\s*station)\b\s*[:\-]?\s*([^\n]+)",
        r"(?im)\b(?:part\s*no\.?\s*and\s*name)\b\s*[:\-]?\s*([^\n]+)",
    ]

    for pattern in constituency_patterns:
        match = re.search(pattern, text)
        if match:
            constituency = " ".join(match.group(1).split())[:200]
            break

    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            section_name = " ".join(match.group(1).split())[:200]
            break

    return constituency, section_name


def _split_candidate_blocks(text: str) -> list[str]:
    cleaned_text = text.replace("\r", "\n")
    if not cleaned_text.strip():
        return []

    blocks: list[str] = []
    primary_blocks = re.split(r"\n\s*\n+", cleaned_text)
    for primary in primary_blocks:
        candidate = primary.strip()
        if not candidate:
            continue
        serial_splits = re.split(r"(?m)(?=^\s*\d{1,4}\b)", candidate)
        for segment in serial_splits:
            normalized_segment = segment.strip()
            if normalized_segment:
                blocks.append(normalized_segment)

    if not blocks:
        return [cleaned_text.strip()]
    return blocks


def _extract_first(patterns: Iterable[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return " ".join(match.group(1).split())
    return ""


def _infer_name(block: str, elector_id: str) -> str:
    explicit_name = _extract_first(
        [
            r"\b(?:name|elector'?s\s+name)\b\s*[:\-]?\s*([A-Za-z][A-Za-z .']{1,80})",
            r"\b(?:नाम)\b\s*[:\-]?\s*([^\n]{2,80})",
        ],
        block,
    )
    if explicit_name:
        return explicit_name[:120]

    for line in block.splitlines():
        value = line.strip()
        if not value:
            continue
        if elector_id and elector_id in value:
            continue
        if re.match(r"^\d{1,4}\b", value):
            continue
        if re.search(r"\b(?:age|gender|house|father|mother|husband|elector)\b", value, re.IGNORECASE):
            continue
        alpha_ratio = sum(ch.isalpha() for ch in value) / max(len(value), 1)
        if alpha_ratio > 0.45:
            return value[:120]
    return ""


def _render_pdf_page_image(file_path: str, page_number: int):
    try:
        import fitz
    except Exception as exc:  # noqa: BLE001
        return None, f"PyMuPDF import failed: {exc}"

    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        return None, f"Pillow import failed: {exc}"

    pdf_doc = None
    try:
        pdf_doc = fitz.open(file_path)
        if getattr(pdf_doc, "needs_pass", False):
            try:
                unlocked = pdf_doc.authenticate("")
            except Exception:  # noqa: BLE001
                unlocked = False
            if not unlocked:
                return None, "password-protected pdf"

        page_index = int(page_number) - 1
        if page_index < 0 or page_index >= int(pdf_doc.page_count):
            return None, f"page {page_number} out of range"

        page = pdf_doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        with Image.open(io.BytesIO(pix.tobytes("png"))) as image:
            return image.convert("RGB"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"pdf render failed: {exc}"
    finally:
        if pdf_doc is not None:
            try:
                pdf_doc.close()
            except Exception:  # noqa: BLE001
                pass


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed_value = float(raw_value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed_value))


def _build_grid_card_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    cols = 3
    margin_x = int(width * 0.03)
    top_margin = int(height * 0.18)
    bottom_margin = int(height * 0.03)
    gap_x = max(4, int(width * 0.008))

    usable_width = max(10, width - (2 * margin_x) - (gap_x * (cols - 1)))
    box_width = max(10, usable_width // cols)
    estimated_box_height = max(10, int(box_width / 2.0))
    usable_height = max(10, height - top_margin - bottom_margin)
    rows = max(7, min(11, int(round(usable_height / max(estimated_box_height, 1)))))
    gap_y = max(4, int(height * 0.005))
    usable_height_with_gaps = max(10, usable_height - (gap_y * (rows - 1)))
    box_height = max(10, usable_height_with_gaps // rows)

    boxes: list[tuple[int, int, int, int]] = []
    for row_index in range(rows):
        y1 = top_margin + row_index * (box_height + gap_y)
        y2 = min(height - bottom_margin, y1 + box_height)
        for col_index in range(cols):
            x1 = margin_x + col_index * (box_width + gap_x)
            x2 = min(width - margin_x, x1 + box_width)
            boxes.append((x1, y1, x2, y2))
    return boxes


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = max(1, area_a + area_b - intersection)
    return intersection / union


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    sorted_boxes = sorted(
        boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        reverse=True,
    )
    kept_boxes: list[tuple[int, int, int, int]] = []
    for candidate in sorted_boxes:
        if any(_box_iou(candidate, existing) > 0.65 for existing in kept_boxes):
            continue
        kept_boxes.append(candidate)
    kept_boxes.sort(key=lambda box: (box[1], box[0]))
    return kept_boxes


def _expand_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    pad_left_ratio = _float_env("NAME_SEARCH_CARD_PAD_LEFT", 0.02, minimum=0.0, maximum=0.1)
    pad_right_ratio = _float_env("NAME_SEARCH_CARD_PAD_RIGHT", 0.02, minimum=0.0, maximum=0.1)
    pad_top_ratio = _float_env("NAME_SEARCH_CARD_PAD_TOP", 0.03, minimum=0.0, maximum=0.12)
    pad_bottom_ratio = _float_env("NAME_SEARCH_CARD_PAD_BOTTOM", 0.05, minimum=0.0, maximum=0.15)
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    expanded_x1 = max(0, int(round(x1 - (box_width * pad_left_ratio))))
    expanded_x2 = min(width, int(round(x2 + (box_width * pad_right_ratio))))
    expanded_y1 = max(0, int(round(y1 - (box_height * pad_top_ratio))))
    expanded_y2 = min(height, int(round(y2 + (box_height * pad_bottom_ratio))))
    return expanded_x1, expanded_y1, expanded_x2, expanded_y2


def _filter_boxes_by_area(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 6:
        return boxes
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    median_area = median(areas)
    min_area = median_area * 0.45
    max_area = median_area * 2.4
    return [box for box, area in zip(boxes, areas) if min_area <= area <= max_area]


def _kmeans_1d(values: Sequence[float], cluster_count: int, max_iter: int = 12) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) <= cluster_count:
        return sorted_values

    centers = []
    last_index = len(sorted_values) - 1
    for cluster_index in range(cluster_count):
        quantile = cluster_index / max(cluster_count - 1, 1)
        seed_index = int(round(quantile * last_index))
        centers.append(sorted_values[seed_index])

    for _ in range(max_iter):
        groups: list[list[float]] = [[] for _ in range(cluster_count)]
        for value in sorted_values:
            nearest = min(range(cluster_count), key=lambda idx: abs(value - centers[idx]))
            groups[nearest].append(value)
        new_centers = []
        for index, group in enumerate(groups):
            if group:
                new_centers.append(sum(group) / len(group))
            else:
                new_centers.append(centers[index])
        shift = max(abs(new_centers[index] - centers[index]) for index in range(cluster_count))
        centers = new_centers
        if shift < 1.0:
            break
    return sorted(centers)


def _compress_column_duplicates(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(boxes) < 10:
        return boxes
    heights = [max(1, box[3] - box[1]) for box in boxes]
    median_height = median(heights)
    x_centers = [((box[0] + box[2]) / 2.0) for box in boxes]
    column_centers = _kmeans_1d(x_centers, cluster_count=3)
    if len(column_centers) < 3:
        return boxes

    columns: list[list[tuple[int, int, int, int]]] = [[] for _ in range(3)]
    for box in boxes:
        center_x = (box[0] + box[2]) / 2.0
        nearest_col = min(range(3), key=lambda idx: abs(center_x - column_centers[idx]))
        columns[nearest_col].append(box)

    compressed: list[tuple[int, int, int, int]] = []
    y_merge_threshold = max(8, int(round(median_height * 0.5)))
    for column_boxes in columns:
        if not column_boxes:
            continue
        sorted_by_y = sorted(column_boxes, key=lambda box: (box[1] + box[3]) / 2.0)
        groups: list[list[tuple[int, int, int, int]]] = []
        for box in sorted_by_y:
            center_y = (box[1] + box[3]) / 2.0
            if not groups:
                groups.append([box])
                continue
            last_group = groups[-1]
            last_center_y = (last_group[-1][1] + last_group[-1][3]) / 2.0
            if abs(center_y - last_center_y) <= y_merge_threshold:
                last_group.append(box)
            else:
                groups.append([box])
        for group in groups:
            largest_box = max(group, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
            compressed.append(largest_box)
    return compressed


def _extract_boxes_from_mask(
    mask_image,
    *,
    width: int,
    height: int,
    cv2_module,
) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2_module.findContours(mask_image, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
    page_area = max(1, width * height)
    candidate_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2_module.boundingRect(contour)
        area = box_width * box_height
        if area < page_area * 0.0018:
            continue
        if area > page_area * 0.11:
            continue
        if box_width < width * 0.16 or box_width > width * 0.45:
            continue
        if box_height < height * 0.045 or box_height > height * 0.20:
            continue
        if y < int(height * 0.10):
            continue
        ratio = box_width / max(box_height, 1)
        if ratio < 0.95 or ratio > 4.8:
            continue
        candidate_boxes.append((x, y, x + box_width, y + box_height))
    return candidate_boxes


def _detect_voter_card_boxes(page_image) -> tuple[list[tuple[int, int, int, int]], str, str | None]:
    width, height = page_image.size
    fallback_boxes = _build_grid_card_boxes(width, height)
    cv2_module, np_module, cv2_error = _load_cv2_numpy()
    if cv2_module is None or np_module is None:
        return (
            fallback_boxes,
            "grid_fallback",
            f"opencv unavailable: {cv2_error}. Install dependencies in the active environment (`pip install -r requirements.txt`).",
        )

    try:
        gray_image = np_module.array(page_image.convert("L"))
        blurred = cv2_module.GaussianBlur(gray_image, (5, 5), 0)
        thresholded = cv2_module.adaptiveThreshold(
            blurred,
            255,
            cv2_module.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2_module.THRESH_BINARY_INV,
            35,
            12,
        )

        close_kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (7, 7))
        closed_mask = cv2_module.morphologyEx(thresholded, cv2_module.MORPH_CLOSE, close_kernel, iterations=1)

        horizontal_kernel = cv2_module.getStructuringElement(
            cv2_module.MORPH_RECT,
            (max(25, int(width * 0.08)), 1),
        )
        vertical_kernel = cv2_module.getStructuringElement(
            cv2_module.MORPH_RECT,
            (1, max(25, int(height * 0.04))),
        )
        horizontal_lines = cv2_module.morphologyEx(thresholded, cv2_module.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2_module.morphologyEx(thresholded, cv2_module.MORPH_OPEN, vertical_kernel, iterations=1)
        grid_mask = cv2_module.bitwise_or(horizontal_lines, vertical_lines)
        combined_mask = cv2_module.bitwise_or(closed_mask, grid_mask)

        contour_candidates = _extract_boxes_from_mask(
            combined_mask,
            width=width,
            height=height,
            cv2_module=cv2_module,
        )
        line_candidates = _extract_boxes_from_mask(
            grid_mask,
            width=width,
            height=height,
            cv2_module=cv2_module,
        )
    except Exception as exc:  # noqa: BLE001
        return fallback_boxes, "grid_fallback", f"opencv detection failed: {exc}"

    candidate_boxes = _dedupe_boxes(contour_candidates + line_candidates)
    candidate_boxes = _filter_boxes_by_area(candidate_boxes)
    candidate_boxes = _compress_column_duplicates(candidate_boxes)
    candidate_boxes = _dedupe_boxes(candidate_boxes)

    if 12 <= len(candidate_boxes) <= 42:
        expanded_boxes = [
            _expand_box(box, width=width, height=height)
            for box in candidate_boxes
        ]
        return expanded_boxes, "opencv_layout", None
    return fallback_boxes, "grid_fallback", "opencv detected unstable card layout; using grid fallback"


def _preprocess_card_for_ocr(card_image):
    from PIL import Image

    cv2_module, np_module, _ = _load_cv2_numpy()
    source_image = card_image.convert("RGB")
    if cv2_module is None or np_module is None:
        gray_image = source_image.convert("L")
        scale_factor = 2.0 if gray_image.width < 900 else 1.5 if gray_image.width < 1200 else 1.0
        if scale_factor > 1.0:
            gray_image = gray_image.resize(
                (
                    int(gray_image.width * scale_factor),
                    int(gray_image.height * scale_factor),
                ),
                Image.Resampling.LANCZOS,
            )
        return gray_image, f"pil_gray_resize_x{scale_factor:.1f}"

    rgb_array = np_module.array(source_image)
    gray_array = cv2_module.cvtColor(rgb_array, cv2_module.COLOR_RGB2GRAY)

    scale_factor = _float_env("NAME_SEARCH_OCR_SCALE", 1.6, minimum=1.0, maximum=3.0)
    if gray_array.shape[1] >= 1400:
        scale_factor = min(scale_factor, 1.2)
    elif gray_array.shape[1] >= 1000:
        scale_factor = min(scale_factor, 1.4)
    if scale_factor > 1.0:
        gray_array = cv2_module.resize(
            gray_array,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2_module.INTER_CUBIC,
        )

    denoised = cv2_module.bilateralFilter(gray_array, 7, 45, 45)
    thresholded = cv2_module.adaptiveThreshold(
        denoised,
        255,
        cv2_module.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2_module.THRESH_BINARY,
        31,
        8,
    )
    open_kernel = cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (2, 2))
    cleaned = cv2_module.morphologyEx(thresholded, cv2_module.MORPH_OPEN, open_kernel, iterations=1)
    return Image.fromarray(cleaned), f"opencv_gray_adaptivethresh_x{scale_factor:.1f}"


def _ocr_text_score(text: str) -> tuple[int, int]:
    normalized = str(text or "")
    return (
        sum(char.isalnum() for char in normalized),
        len(normalized.strip()),
    )


def _run_tesseract(image, *, timeout_seconds: float | None, config: str):
    import pytesseract

    if timeout_seconds is not None and timeout_seconds > 0:
        return pytesseract.image_to_string(
            image,
            timeout=float(timeout_seconds),
            config=config,
        ) or ""
    return pytesseract.image_to_string(
        image,
        config=config,
    ) or ""


def _ocr_card_image(card_image, timeout_seconds: float | None) -> tuple[str, str | None, dict[str, Any]]:
    try:
        import pytesseract
    except Exception as exc:  # noqa: BLE001
        return "", f"pytesseract import failed: {exc}", {"preprocess": "none"}

    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return "", "tesseract command not found", {"preprocess": "none"}

    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except Exception:  # noqa: BLE001
        pass

    preprocessed_image, preprocess_note = _preprocess_card_for_ocr(card_image)
    ocr_meta = {
        "preprocess": preprocess_note,
        "attempts": [],
    }

    try:
        processed_text = _run_tesseract(preprocessed_image, timeout_seconds=timeout_seconds, config="--oem 1 --psm 6")
        raw_text = _run_tesseract(card_image, timeout_seconds=timeout_seconds, config="--oem 1 --psm 11")
        ocr_meta["attempts"] = [
            {"source": "preprocessed", "config": "psm6", "score": _ocr_text_score(processed_text)},
            {"source": "raw", "config": "psm11", "score": _ocr_text_score(raw_text)},
        ]
        text = processed_text if _ocr_text_score(processed_text) >= _ocr_text_score(raw_text) else raw_text
    except RuntimeError as exc:
        if "time" in str(exc).lower():
            timeout_display = int(timeout_seconds) if timeout_seconds else "configured"
            return "", f"ocr timeout after {timeout_display}s", ocr_meta
        return "", f"ocr failed: {exc}", ocr_meta
    except Exception as exc:  # noqa: BLE001
        return "", f"ocr failed: {exc}", ocr_meta

    return text, None, ocr_meta


def _clean_card_field(raw_value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(raw_value or "")).strip()
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
    cleaned = re.sub(
        r"(?i)\b(?:father|mother|husband|wife)(?:'?s)?\s+name\b.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?:house|gender|age|sex|epic|elector|constituency|section|photo|available|poc|relation)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" :|-.,")
    return cleaned[:160]


def _clean_card_ocr_text(raw_text: str) -> str:
    normalized_text = str(raw_text or "").replace("\r", "\n")
    cleaned_lines: list[str] = []
    for raw_line in normalized_text.splitlines():
        line = raw_line
        line = line.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", "'")
        line = line.replace("¦", "|")
        line = re.sub(r"[#?+]+", " ", line)
        line = re.sub(r"\s+", " ", line).strip(" -:|")
        if not line:
            continue
        lowered = line.lower()
        if any(
            noise in lowered
            for noise in [
                "photo",
                "available",
                "poc",
                "assembly constituency",
                "constituency no",
                "section name",
                "part no",
                "electoral roll",
            ]
        ):
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue
        cleaned_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def _extract_labeled_value(cleaned_text: str, label_patterns: Sequence[str]) -> str:
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        for label_pattern in label_patterns:
            label_match = re.search(
                rf"(?i)\b{label_pattern}\b(?:\s*name)?\s*[:\-]?\s*(.*)$",
                line,
            )
            if not label_match:
                continue
            immediate_value = _clean_card_field(label_match.group(1))
            if immediate_value:
                return immediate_value
            if index + 1 < len(lines):
                next_line_value = _clean_card_field(lines[index + 1])
                if next_line_value:
                    return next_line_value
    return ""


def _parse_serial_number(cleaned_text: str) -> str:
    labeled_serial = _extract_labeled_value(
        cleaned_text,
        [r"serial", r"sl\.?\s*no\.?", r"sr\.?\s*no\.?", r"क्रमांक"],
    )
    if labeled_serial and re.search(r"\d", labeled_serial):
        match = re.search(r"\b(\d{1,5})\b", labeled_serial)
        if match:
            return match.group(1)

    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    for line in lines:
        if any(token in line.lower() for token in ["age", "gender", "sex", "house", "father", "mother", "husband"]):
            continue
        serial_with_epic = re.search(
            r"^\s*(\d{1,5})\s+(?:[A-Z]{2,4}\d{6,10}|[A-Z]{1,3}/\d{1,3}/\d{1,3}/\d{3,8}|[A-Z]{1,4}[-/]\d{6,10})\b",
            line.upper(),
        )
        if serial_with_epic:
            return serial_with_epic.group(1)

    if lines:
        first_line_match = re.match(r"^(\d{1,5})\b", lines[0])
        if first_line_match:
            return first_line_match.group(1)
    for line in lines:
        if re.fullmatch(r"\d{1,5}", line):
            return line
    return ""


def _parse_elector_id(cleaned_text: str) -> str:
    labeled_id = _extract_labeled_value(
        cleaned_text,
        [r"epic", r"elector\s*id", r"voter\s*id", r"id\s*no\.?", r"card\s*no\.?"],
    )
    if labeled_id:
        normalized_labeled = re.sub(r"\s+", "", labeled_id.upper())
        if re.fullmatch(r"(?:[A-Z]{2,4}\d{6,10}|[A-Z]{1,3}/\d{1,3}/\d{1,3}/\d{3,8}|[A-Z]{1,4}[-/]\d{6,10})", normalized_labeled):
            return normalized_labeled

    upper_text = cleaned_text.upper()
    patterns = [
        r"([A-Z]{2,4}\s*\d{6,10})",
        r"([A-Z]{1,3}\s*/\s*\d{1,3}\s*/\s*\d{1,3}\s*/\s*\d{3,8})",
        r"([A-Z]{1,4}\s*[-/]\s*\d{6,10})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, upper_text):
            candidate = match.group(1)
            normalized = re.sub(r"\s+", "", candidate)
            normalized = re.sub(r"\s*/\s*", "/", normalized)
            normalized = re.sub(r"\s*-\s*", "-", normalized)
            digit_count = sum(char.isdigit() for char in normalized)
            if digit_count >= 6:
                return normalized
    return ""


def _parse_relative_fields(cleaned_text: str) -> tuple[str, str]:
    relative_patterns: list[tuple[str, str]] = [
        ("father", r"father(?:'?s)?"),
        ("husband", r"husband(?:'?s)?"),
        ("mother", r"mother(?:'?s)?"),
        ("husband", r"wife(?:'?s)?"),
    ]
    for normalized_type, pattern in relative_patterns:
        relative_name = _extract_labeled_value(cleaned_text, [pattern])
        if relative_name:
            return normalized_type, relative_name
        for line in cleaned_text.splitlines():
            reverse_match = re.search(
                rf"(?i)^\s*([A-Za-z][A-Za-z .']{{1,120}}?)\s+{pattern}\s*(?:name)?\s*$",
                line.strip(),
            )
            if reverse_match:
                candidate = _clean_card_field(reverse_match.group(1))
                if candidate:
                    return normalized_type, candidate

    shorthand_match = re.search(
        r"(?im)\b(S/O|W/O|D/O)\b\s*[:\-]?\s*([^\n]{2,120})",
        cleaned_text,
    )
    if shorthand_match:
        token = shorthand_match.group(1).upper()
        relative_name = _clean_card_field(shorthand_match.group(2))
        if token == "S/O":
            return "father", relative_name
        if token == "W/O":
            return "husband", relative_name
        if token == "D/O":
            return "mother", relative_name
    return "", ""


def _parse_house_number(cleaned_text: str) -> str:
    house_value = _extract_labeled_value(
        cleaned_text,
        [r"house\s*(?:no\.?|number)?", r"h\.?\s*no\.?", r"house"],
    )
    if house_value:
        return _clean_card_field(house_value)

    fallback_match = re.search(r"(?im)\b(?:h\.?\s*no\.?|house)\b\W*([A-Za-z0-9/\- ]{1,40})", cleaned_text)
    if fallback_match:
        return _clean_card_field(fallback_match.group(1))
    shorthand_match = re.search(r"(?im)\bHN\b\W*([A-Za-z0-9/\- ]{1,40})", cleaned_text)
    if shorthand_match:
        return _clean_card_field(shorthand_match.group(1))
    return ""


def _parse_age_and_gender(cleaned_text: str) -> tuple[int | None, str]:
    age = None
    age_match = re.search(r"(?im)\b(?:age)\b\s*[:\-]?\s*(\d{1,3})", cleaned_text)
    if age_match and age_match.group(1).isdigit():
        age = int(age_match.group(1))
    if age is None:
        fallback_age = re.search(r"(?im)\b(\d{1,3})\b\s*(?:yrs?|years?)\b", cleaned_text)
        if fallback_age and fallback_age.group(1).isdigit():
            age = int(fallback_age.group(1))
    if age is None:
        fallback_age = re.search(r"(?im)\b(\d{1,3})\s*/\s*(male|female|other|m|f)\b", cleaned_text)
        if fallback_age and fallback_age.group(1).isdigit():
            age = int(fallback_age.group(1))
    if age is None:
        fallback_age = re.search(r"(?im)\b(?:male|female|other|m|f)\s*/\s*(\d{1,3})\b", cleaned_text)
        if fallback_age and fallback_age.group(1).isdigit():
            age = int(fallback_age.group(1))

    gender = ""
    gender_match = re.search(r"(?im)\b(?:gender|sex)\b\s*[:\-]?\s*(male|female|other|m|f)\b", cleaned_text)
    if not gender_match:
        gender_match = re.search(r"(?im)\b(male|female|other)\b", cleaned_text)
    if gender_match:
        gender = gender_match.group(1).lower()
    if not gender:
        fallback_gender = re.search(r"(?im)\b(\d{1,3})\s*/\s*(male|female|other|m|f)\b", cleaned_text)
        if fallback_gender:
            gender = fallback_gender.group(2).lower()
    if not gender:
        fallback_gender = re.search(r"(?im)\b(?:male|female|other|m|f)\s*/\s*(\d{1,3})\b", cleaned_text)
        if fallback_gender:
            gender = fallback_gender.group(0).split("/")[0].strip().lower()
    if gender == "m":
        gender = "male"
    elif gender == "f":
        gender = "female"
    return age, gender


def _is_noise_name(name_value: str) -> bool:
    name = _clean_card_field(name_value).lower()
    if not name:
        return True
    if name in {"photo", "available", "poc"}:
        return True
    if any(
        token in name
        for token in [
            "assembly constituency",
            "constituency no",
            "section name",
            "part no",
            "electoral roll",
            "hous",
            "gender",
            "age",
            "father name",
            "husband name",
            "mother name",
        ]
    ):
        return True
    alpha_chars = sum(char.isalpha() for char in name)
    return alpha_chars < 3


def _infer_card_name(cleaned_text: str, elector_id: str) -> str:
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    for line in lines:
        lower_line = line.lower()
        if elector_id and elector_id in line.replace(" ", ""):
            continue
        if re.match(r"^\d{1,5}\b", line):
            continue
        if any(
            token in lower_line
            for token in [
                "father",
                "mother",
                "husband",
                "wife",
                "house",
                "gender",
                "age",
                "sex",
                "epic",
                "elector",
                "serial",
                "sl no",
            ]
        ):
            continue
        candidate = _clean_card_field(line)
        if _is_noise_name(candidate):
            continue
        alpha_ratio = sum(char.isalpha() for char in candidate) / max(len(candidate), 1)
        if alpha_ratio >= 0.6:
            return candidate[:120]
    return ""


def _classify_card_record(record: dict[str, Any]) -> tuple[str, str]:
    name_value = str(record.get("name") or "").strip()
    if not name_value or _is_noise_name(name_value):
        return "rejected", "invalid/noisy name"

    elector_id_value = str(record.get("elector_id") or "").strip()
    if elector_id_value and not re.fullmatch(
        r"(?:[A-Z]{2,4}\d{6,10}|[A-Z]{1,3}/\d{1,3}/\d{1,3}/\d{3,8}|[A-Z]{1,4}[-/]\d{6,10})",
        elector_id_value,
    ):
        return "rejected", "invalid elector_id format"

    serial_number_value = str(record.get("serial_number") or "").strip()
    if serial_number_value and not re.fullmatch(r"\d{1,5}", serial_number_value):
        return "rejected", "invalid serial_number format"

    primary_fields_present = sum(
        [
            1 if elector_id_value else 0,
            1 if serial_number_value else 0,
            1 if str(record.get("house_number") or "").strip() else 0,
            1 if record.get("age") is not None else 0,
        ]
    )
    secondary_fields_present = sum(
        [
            1 if str(record.get("relative_name") or "").strip() else 0,
            1 if str(record.get("gender") or "").strip() else 0,
        ]
    )

    if primary_fields_present >= 2:
        return "valid", ""
    if primary_fields_present >= 1:
        return "partial", "missing some key fields"
    if secondary_fields_present >= 1:
        return "partial", "only secondary fields parsed"
    return "rejected", "missing minimum identity fields"


def _parse_card_record(
    *,
    raw_card_text: str,
    cleaned_card_text: str,
    file_name: str,
    file_path: str,
    page_number: int,
    constituency: str | None,
    section_name: str | None,
    extraction_method: str,
) -> tuple[dict[str, Any] | None, str, str | None]:
    if not cleaned_card_text.strip():
        return None, "rejected", "empty card text after cleanup"

    serial_number = _parse_serial_number(cleaned_card_text)
    elector_id = _parse_elector_id(cleaned_card_text)
    name = _extract_labeled_value(
        cleaned_card_text,
        [
            r"name of elector",
            r"elector'?s name",
            r"name",
            r"नाम",
        ],
    )
    if not name:
        name = _infer_card_name(cleaned_card_text, elector_id)
    name = re.sub(
        r"(?i)\b(?:father|mother|husband|wife)(?:'?s)?\s+name\b.*$",
        "",
        name,
    )
    name = _clean_card_field(name)

    relative_type, relative_name = _parse_relative_fields(cleaned_card_text)
    relative_name = _clean_card_field(relative_name)
    house_number = _parse_house_number(cleaned_card_text)
    age, gender = _parse_age_and_gender(cleaned_card_text)

    parsed_record = {
        "serial_number": serial_number or None,
        "elector_id": elector_id or None,
        "name": name or None,
        "relative_name": relative_name or None,
        "relative_type": relative_type or None,
        "house_number": house_number or None,
        "age": age,
        "gender": gender or None,
        "constituency": constituency or None,
        "section_name": section_name or None,
        "file_name": file_name,
        "file_path": file_path,
        "page_number": int(page_number),
        "extraction_method": extraction_method,
        "raw_record_text": raw_card_text.replace("\r", "\n").strip(),
    }
    parse_status, reason = _classify_card_record(parsed_record)
    if parse_status == "rejected":
        return None, parse_status, reason
    parsed_record["_parse_status"] = parse_status
    return parsed_record, parse_status, reason or None


def parse_voter_records_from_page_layout_aware(
    *,
    page_text: str,
    file_name: str,
    file_path: str,
    page_number: int,
    extraction_method: str,
    ocr_timeout_seconds: float | None = 20.0,
    max_preview_cards: int = 5,
    include_card_debug: bool = False,
) -> dict[str, Any]:
    constituency, section_name = _extract_context(page_text)
    fallback_records = parse_voter_records_from_page(
        page_text=page_text,
        file_name=file_name,
        file_path=file_path,
        page_number=page_number,
        extraction_method=extraction_method,
    )

    debug_payload: dict[str, Any] = {
        "mode": "card_ocr_layout",
        "detection_strategy": "",
        "detection_error": None,
        "cards_detected": 0,
        "cards_with_text": 0,
        "cards_parsed": 0,
        "cards_valid": 0,
        "cards_partial": 0,
        "cards_rejected": 0,
        "cards_inserted": 0,
        "expected_card_count": 0,
        "reject_reason_breakdown": {},
        "cards": [],
    }

    page_image, render_error = _render_pdf_page_image(file_path=file_path, page_number=page_number)
    if page_image is None:
        debug_payload["mode"] = "page_text_fallback"
        debug_payload["detection_error"] = render_error or "page image unavailable"
        debug_payload["cards_parsed"] = len(fallback_records)
        debug_payload["cards_valid"] = len(fallback_records)
        debug_payload["cards_partial"] = 0
        return {"records": fallback_records, "debug": debug_payload}

    boxes, detection_strategy, detection_error = _detect_voter_card_boxes(page_image)
    debug_payload["detection_strategy"] = detection_strategy
    debug_payload["detection_error"] = detection_error
    debug_payload["cards_detected"] = len(boxes)
    if len(boxes) >= 18:
        debug_payload["expected_card_count"] = 24
    else:
        debug_payload["expected_card_count"] = len(boxes)
    if not boxes:
        debug_payload["mode"] = "page_text_fallback"
        debug_payload["cards_parsed"] = len(fallback_records)
        debug_payload["cards_valid"] = len(fallback_records)
        debug_payload["cards_partial"] = 0
        debug_payload["cards_rejected"] = max(0, int(debug_payload["cards_detected"]) - int(debug_payload["cards_parsed"]))
        return {"records": fallback_records, "debug": debug_payload}

    if not constituency or not section_name:
        header_crop = page_image.crop((0, 0, page_image.size[0], max(1, int(page_image.size[1] * 0.2))))
        header_text, _, _ = _ocr_card_image(header_crop, timeout_seconds=min(float(ocr_timeout_seconds or 20), 8.0))
        header_constituency, header_section = _extract_context(header_text)
        constituency = constituency or header_constituency or None
        section_name = section_name or header_section or None

    parsed_records: list[dict[str, Any]] = []
    cards_debug: list[dict[str, Any]] = []
    for card_index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        card_image = page_image.crop((x1, y1, x2, y2))
        raw_card_text, ocr_error, ocr_meta = _ocr_card_image(card_image, timeout_seconds=ocr_timeout_seconds)
        cleaned_card_text = _clean_card_ocr_text(raw_card_text)
        if cleaned_card_text.strip():
            debug_payload["cards_with_text"] = int(debug_payload["cards_with_text"]) + 1

        record, parse_status, reject_reason = _parse_card_record(
            raw_card_text=raw_card_text,
            cleaned_card_text=cleaned_card_text,
            file_name=file_name,
            file_path=file_path,
            page_number=page_number,
            constituency=constituency,
            section_name=section_name,
            extraction_method=f"card_ocr_{detection_strategy}",
        )
        accepted = parse_status in {"valid", "partial"} and record is not None
        if accepted:
            parsed_records.append(record)
            if parse_status == "valid":
                debug_payload["cards_valid"] = int(debug_payload.get("cards_valid", 0)) + 1
            else:
                debug_payload["cards_partial"] = int(debug_payload.get("cards_partial", 0)) + 1
        else:
            debug_payload["cards_rejected"] = int(debug_payload.get("cards_rejected", 0)) + 1
            reason_key = (reject_reason or "rejected").strip() or "rejected"
            reject_breakdown = dict(debug_payload.get("reject_reason_breakdown", {}))
            reject_breakdown[reason_key] = int(reject_breakdown.get(reason_key, 0)) + 1
            debug_payload["reject_reason_breakdown"] = reject_breakdown

        if include_card_debug:
            card_debug_entry: dict[str, Any] = {
                "card_index": card_index,
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "ocr_error": ocr_error,
                "ocr_preprocess": (ocr_meta or {}).get("preprocess"),
                "accepted": accepted,
                "reject_reason": reject_reason,
                "parse_status": parse_status,
                "validation_status": parse_status,
                "db_insert_status": "pending_insert" if accepted else "rejected_not_inserted",
                "raw_ocr_text": raw_card_text.strip(),
                "cleaned_ocr_text": cleaned_card_text,
                "parsed_record": record,
            }
            if card_index <= max(1, int(max_preview_cards)):
                crop_buffer = io.BytesIO()
                try:
                    card_image.save(crop_buffer, format="PNG")
                    card_debug_entry["crop_png_bytes"] = crop_buffer.getvalue()
                except Exception:  # noqa: BLE001
                    card_debug_entry["crop_png_bytes"] = None
            cards_debug.append(card_debug_entry)

    debug_payload["cards_parsed"] = len(parsed_records)
    debug_payload["cards_valid"] = int(debug_payload.get("cards_valid", 0))
    debug_payload["cards_partial"] = int(debug_payload.get("cards_partial", 0))
    if "reject_reason_breakdown" not in debug_payload:
        debug_payload["reject_reason_breakdown"] = {}
    debug_payload["cards"] = cards_debug
    if not parsed_records and not debug_payload.get("detection_error"):
        debug_payload["detection_error"] = "no valid card records parsed from detected card regions"
    return {"records": parsed_records, "debug": debug_payload}


def parse_voter_records_from_page(
    *,
    page_text: str,
    file_name: str,
    file_path: str,
    page_number: int,
    extraction_method: str,
) -> list[dict[str, Any]]:
    if not page_text.strip():
        return []

    constituency, section_name = _extract_context(page_text)
    blocks = _split_candidate_blocks(page_text)
    parsed_records: list[dict[str, Any]] = []

    for block in blocks:
        elector_id = _extract_first([r"\b([A-Z]{2,4}\d{6,12})\b"], block)
        serial_number = _extract_first(
            [
                r"^\s*(\d{1,5})\b",
                r"\b(?:serial|sl\.?\s*no\.?|क्रमांक)\b\s*[:\-]?\s*(\d{1,5})",
            ],
            block,
        )
        relative_match = re.search(
            r"\b(Father|Mother|Husband|Wife|S/O|D/O|W/O)\b\s*[:\-]?\s*([A-Za-z .']{2,120})",
            block,
            flags=re.IGNORECASE,
        )
        relative_type = ""
        relative_name = ""
        if relative_match:
            relative_type_raw = relative_match.group(1).strip().lower()
            if relative_type_raw in {"s/o", "father"}:
                relative_type = "father"
            elif relative_type_raw in {"w/o", "husband", "wife"}:
                relative_type = "husband"
            elif relative_type_raw in {"d/o", "mother"}:
                relative_type = "mother"
            else:
                relative_type = relative_type_raw
            relative_name = " ".join(relative_match.group(2).split())[:120]

        house_number = _extract_first(
            [
                r"\b(?:house\s*no\.?|h\.?\s*no\.?)\b\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
                r"\b(?:house)\b\s*[:\-]?\s*([A-Za-z0-9/\-]+)",
            ],
            block,
        )

        age_value = _extract_first([r"\b(?:age)\b\s*[:\-]?\s*(\d{1,3})"], block)
        age = int(age_value) if age_value.isdigit() else None

        gender = _extract_first(
            [
                r"\b(?:gender|sex)\b\s*[:\-]?\s*(male|female|other|m|f)",
                r"\b(male|female|other)\b",
            ],
            block,
        )
        gender = gender.lower()
        if gender == "m":
            gender = "male"
        elif gender == "f":
            gender = "female"

        name = _infer_name(block, elector_id)
        if not any([name, elector_id, relative_name, house_number]):
            continue

        parsed_records.append(
            {
                "serial_number": serial_number or None,
                "elector_id": elector_id or None,
                "name": name or None,
                "relative_name": relative_name or None,
                "relative_type": relative_type or None,
                "house_number": house_number or None,
                "age": age,
                "gender": gender or None,
                "constituency": constituency or None,
                "section_name": section_name or None,
                "file_name": file_name,
                "file_path": file_path,
                "page_number": int(page_number),
                "extraction_method": extraction_method or "",
                "raw_record_text": block.strip(),
            }
        )

    return parsed_records
