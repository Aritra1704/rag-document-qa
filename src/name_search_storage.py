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


def fetch_document_row_by_file_path(connection, *, file_path: str) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            id,
            folder_path,
            file_name,
            file_path,
            pages_total,
            pages_processed,
            status,
            error_message,
            last_modified,
            file_size,
            last_processed_at,
            created_at,
            updated_at
        FROM documents
        WHERE file_path = %s
        LIMIT 1
        """,
        (file_path,),
    ).fetchone()
    return dict(row) if row else None


def fetch_document_page_status_map(
    connection,
    *,
    document_id: int,
    start_page: int | None = None,
    end_page: int | None = None,
) -> dict[int, str]:
    filters: list[str] = ["document_id = %s"]
    params: list[Any] = [int(document_id)]
    if start_page is not None:
        filters.append("page_number >= %s")
        params.append(int(start_page))
    if end_page is not None:
        filters.append("page_number <= %s")
        params.append(int(end_page))
    sql_query = (
        "SELECT page_number, status FROM pages WHERE "
        + " AND ".join(filters)
        + " ORDER BY page_number"
    )
    rows = connection.execute(sql_query, tuple(params)).fetchall()
    return {int(row["page_number"]): str(row["status"] or "") for row in rows}


def count_document_pages_by_status(
    connection,
    *,
    document_id: int,
    statuses: Sequence[str],
    start_page: int | None = None,
    end_page: int | None = None,
) -> int:
    normalized_statuses = [str(status or "").strip() for status in statuses if str(status or "").strip()]
    if not normalized_statuses:
        return 0
    filters: list[str] = ["document_id = %s", "status = ANY(%s)"]
    params: list[Any] = [int(document_id), normalized_statuses]
    if start_page is not None:
        filters.append("page_number >= %s")
        params.append(int(start_page))
    if end_page is not None:
        filters.append("page_number <= %s")
        params.append(int(end_page))
    row = connection.execute(
        "SELECT COUNT(*) AS c FROM pages WHERE " + " AND ".join(filters),
        tuple(params),
    ).fetchone()
    return int(row["c"]) if row else 0


def get_ingestion_monitor_summary(connection, *, folder_path: str) -> dict[str, Any]:
    docs_row = connection.execute(
        """
        SELECT
            COUNT(*) AS total_files,
            COUNT(*) FILTER (WHERE status = 'processed') AS completed_files,
            COUNT(*) FILTER (WHERE status = 'processing') AS processing_files,
            COUNT(*) FILTER (WHERE status = 'pending') AS pending_files,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed_files,
            COUNT(*) FILTER (WHERE status = 'skipped') AS skipped_files,
            COALESCE(SUM(pages_total), 0) AS pages_total_expected,
            COALESCE(SUM(pages_processed), 0) AS pages_processed_reported
        FROM documents
        WHERE folder_path = %s
        """,
        (folder_path,),
    ).fetchone()

    pages_row = connection.execute(
        """
        SELECT
            COUNT(*) AS pages_seen,
            COUNT(*) FILTER (WHERE p.status = 'processed') AS pages_completed,
            COUNT(*) FILTER (WHERE p.status = 'processing') AS pages_processing,
            COUNT(*) FILTER (WHERE p.status = 'failed') AS pages_failed,
            COUNT(*) FILTER (WHERE p.status = 'skipped') AS pages_skipped
        FROM pages p
        JOIN documents d ON d.id = p.document_id
        WHERE d.folder_path = %s
        """,
        (folder_path,),
    ).fetchone()

    current_row = connection.execute(
        """
        SELECT
            id,
            file_name,
            file_path,
            pages_processed,
            pages_total,
            status,
            updated_at
        FROM documents
        WHERE folder_path = %s
          AND status = 'processing'
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (folder_path,),
    ).fetchone()

    low_conf_row = connection.execute(
        """
        SELECT COUNT(*) AS c
        FROM parsed_records r
        JOIN documents d ON d.id = r.document_id
        WHERE d.folder_path = %s
          AND (
              r.extraction_method ILIKE '%%|needs_review'
              OR r.extraction_method ILIKE '%%|partial'
          )
        """,
        (folder_path,),
    ).fetchone()

    docs = dict(docs_row) if docs_row else {}
    pages = dict(pages_row) if pages_row else {}
    low_conf = dict(low_conf_row) if low_conf_row else {}

    return {
        "documents": {
            "total_files": int(docs.get("total_files", 0)),
            "completed_files": int(docs.get("completed_files", 0)),
            "processing_files": int(docs.get("processing_files", 0)),
            "pending_files": int(docs.get("pending_files", 0)),
            "failed_files": int(docs.get("failed_files", 0)),
            "skipped_files": int(docs.get("skipped_files", 0)),
            "pages_total_expected": int(docs.get("pages_total_expected", 0)),
            "pages_processed_reported": int(docs.get("pages_processed_reported", 0)),
        },
        "pages": {
            "pages_seen": int(pages.get("pages_seen", 0)),
            "pages_completed": int(pages.get("pages_completed", 0)),
            "pages_processing": int(pages.get("pages_processing", 0)),
            "pages_failed": int(pages.get("pages_failed", 0)),
            "pages_skipped": int(pages.get("pages_skipped", 0)),
        },
        "current_file": dict(current_row) if current_row else None,
        "low_confidence_records": int(low_conf.get("c", 0)),
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


def _normalize_context_value(raw_value: str) -> str:
    value = " ".join(str(raw_value or "").split()).strip(" :-|")
    if not value:
        return ""
    value = re.sub(r"(?i)^no\.?\s+and\s+name\s*[:\-]?\s*", "", value).strip(" :-|")
    value = re.sub(r"(?i)\bpart\s*no\.?\s*[:\-]?\s*\d+\b.*$", "", value).strip(" :-|")
    return value[:200]


def _extract_context(text: str) -> tuple[str, str]:
    constituency = ""
    section_name = ""

    constituency_patterns = [
        r"(?im)^\s*assembly\s+constituency\s*(?:no\.?)?\s*(?:and|&)?\s*name\s*[:\-]\s*([^\n]+)",
        r"(?im)^\s*constituency\s*(?:no\.?)?\s*(?:and|&)?\s*name\s*[:\-]\s*([^\n]+)",
        r"(?im)^\s*(?:assembly\s+constituency|constituency|ac\s*name|constituency\s+name)\s*[:\-]\s*([^\n]+)",
    ]
    section_patterns = [
        r"(?im)^\s*section\s*(?:no\.?)?\s*(?:and|&)?\s*name\s*[:\-]\s*([^\n]+)",
        r"(?im)^\s*(?:section\s*name|part\s*name|polling\s*station)\s*[:\-]\s*([^\n]+)",
        r"(?im)^\s*part\s*no\.?\s*and\s*name\s*[:\-]\s*([^\n]+)",
    ]

    for pattern in constituency_patterns:
        match = re.search(pattern, text or "")
        if match:
            constituency = _normalize_context_value(match.group(1))
            if constituency:
                break

    for pattern in section_patterns:
        match = re.search(pattern, text or "")
        if match:
            section_name = _normalize_context_value(match.group(1))
            if section_name:
                break

    return constituency, section_name


def _extract_part_number(text: str) -> str:
    lines = [line.strip() for line in str(text or "").replace("\r", "\n").splitlines() if line.strip()]
    part_pattern = re.compile(r"(?i)^part\s*no\.?\s*[:\-]?\s*(.*)$")
    for index, line in enumerate(lines):
        match = part_pattern.search(line)
        if not match:
            continue
        immediate_value = _normalize_context_value(match.group(1))
        if immediate_value and re.search(r"\d", immediate_value):
            return immediate_value[:40]
        if index + 1 < len(lines):
            next_line = _normalize_context_value(lines[index + 1])
            if next_line and re.search(r"\d", next_line):
                return next_line[:40]
    fallback_match = re.search(r"(?im)\bpart\s*no\.?\s*[:\-]?\s*([A-Za-z0-9/\-]{1,40})", str(text or ""))
    if fallback_match:
        return _normalize_context_value(fallback_match.group(1))[:40]
    return ""


def _extract_page_header_metadata(
    *,
    page_image,
    support_boxes: Sequence[tuple[int, int, int, int]],
    ocr_timeout_seconds: float | None,
    include_preview: bool,
) -> dict[str, Any]:
    page_width, page_height = page_image.size
    support_top = min(box[1] for box in support_boxes) if support_boxes else int(page_height * 0.24)
    max_header_bottom = int(page_height * 0.30)
    min_header_bottom = int(page_height * 0.12)
    header_bottom = int(round(support_top - max(8, page_height * 0.005)))
    header_bottom = max(min_header_bottom, min(max_header_bottom, header_bottom))
    if header_bottom <= 8:
        header_bottom = max(10, int(page_height * 0.20))

    header_bbox = (0, 0, int(page_width), int(header_bottom))
    header_crop = page_image.crop(header_bbox)
    header_text, header_ocr_error, header_ocr_meta = _ocr_card_image(
        header_crop,
        timeout_seconds=min(float(ocr_timeout_seconds or 20), 8.0),
        fast_mode=True,
    )
    constituency, section_name = _extract_context(header_text)
    part_number = _extract_part_number(header_text)

    header_preview_bytes: bytes | None = None
    if include_preview:
        try:
            preview_buffer = io.BytesIO()
            header_crop.save(preview_buffer, format="PNG")
            header_preview_bytes = preview_buffer.getvalue()
        except Exception:  # noqa: BLE001
            header_preview_bytes = None

    return {
        "bbox": {
            "x1": int(header_bbox[0]),
            "y1": int(header_bbox[1]),
            "x2": int(header_bbox[2]),
            "y2": int(header_bbox[3]),
        },
        "ocr_text": str(header_text or ""),
        "ocr_error": header_ocr_error,
        "ocr_preprocess": (header_ocr_meta or {}).get("preprocess"),
        "crop_png_bytes": header_preview_bytes,
        "metadata": {
            "constituency": constituency or None,
            "section_name": section_name or None,
            "part_number": part_number or None,
        },
    }


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
    rows = int(round(_float_env("NAME_SEARCH_SLOT_ROWS", 10, minimum=8, maximum=12)))
    margin_x = int(width * 0.03)
    top_margin = int(height * 0.18)
    bottom_margin = int(height * 0.03)
    gap_x = max(4, int(width * 0.008))
    gap_y = max(4, int(height * 0.006))

    usable_width = max(10, width - (2 * margin_x) - (gap_x * (cols - 1)))
    usable_height = max(10, height - top_margin - bottom_margin - (gap_y * (rows - 1)))
    box_width = max(10, usable_width // cols)
    box_height = max(10, usable_height // rows)

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
    pad_left_ratio: float | None = None,
    pad_right_ratio: float | None = None,
    pad_top_ratio: float | None = None,
    pad_bottom_ratio: float | None = None,
    extra_top_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    resolved_left = (
        _float_env("NAME_SEARCH_CARD_PAD_LEFT", 0.03, minimum=0.0, maximum=0.15)
        if pad_left_ratio is None
        else max(0.0, min(0.15, float(pad_left_ratio)))
    )
    resolved_right = (
        _float_env("NAME_SEARCH_CARD_PAD_RIGHT", 0.03, minimum=0.0, maximum=0.15)
        if pad_right_ratio is None
        else max(0.0, min(0.15, float(pad_right_ratio)))
    )
    resolved_top = (
        _float_env("NAME_SEARCH_CARD_PAD_TOP", 0.12, minimum=0.0, maximum=0.2)
        if pad_top_ratio is None
        else max(0.0, min(0.2, float(pad_top_ratio)))
    )
    resolved_bottom = (
        _float_env("NAME_SEARCH_CARD_PAD_BOTTOM", 0.05, minimum=0.0, maximum=0.2)
        if pad_bottom_ratio is None
        else max(0.0, min(0.2, float(pad_bottom_ratio)))
    )
    resolved_extra_top = max(0.0, min(0.25, float(extra_top_ratio)))

    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    expanded_x1 = max(0, int(round(x1 - (box_width * resolved_left))))
    expanded_x2 = min(width, int(round(x2 + (box_width * resolved_right))))
    expanded_y1 = max(0, int(round(y1 - (box_height * (resolved_top + resolved_extra_top)))))
    expanded_y2 = min(height, int(round(y2 + (box_height * resolved_bottom))))
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
        return candidate_boxes, "opencv_layout", None
    return fallback_boxes, "grid_fallback", "opencv detected unstable card layout; using grid fallback"


def _group_numeric_values(values: Sequence[float], threshold: float) -> list[list[float]]:
    if not values:
        return []
    sorted_values = sorted(float(value) for value in values)
    groups: list[list[float]] = []
    for value in sorted_values:
        if not groups:
            groups.append([value])
            continue
        if abs(value - groups[-1][-1]) <= threshold:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _estimate_template_rows(support_boxes: Sequence[tuple[int, int, int, int]]) -> int:
    default_rows = int(round(_float_env("NAME_SEARCH_SLOT_ROWS", 10, minimum=8, maximum=12)))
    if not support_boxes:
        return default_rows
    heights = [max(1, box[3] - box[1]) for box in support_boxes]
    median_height = float(median(heights))
    y_centers = [((box[1] + box[3]) / 2.0) for box in support_boxes]
    grouped_rows = _group_numeric_values(y_centers, threshold=max(8.0, median_height * 0.55))
    estimated_rows = len(grouped_rows)
    if estimated_rows < 6:
        return default_rows
    return max(default_rows, min(12, estimated_rows))


def _derive_template_slot_boxes(
    *,
    width: int,
    height: int,
    support_boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    cols = 3
    default_rows = int(round(_float_env("NAME_SEARCH_SLOT_ROWS", 10, minimum=8, maximum=12)))
    default_margin_x = int(width * 0.03)
    default_top = int(height * 0.18)
    default_bottom = int(height * 0.03)

    if not support_boxes:
        rows = default_rows
        gap_x = max(4, int(width * 0.008))
        gap_y = max(4, int(height * 0.006))
        region_x1 = default_margin_x
        region_x2 = width - default_margin_x
        region_y1 = default_top
        region_y2 = height - default_bottom
        region_width = max(10, region_x2 - region_x1)
        region_height = max(10, region_y2 - region_y1)
        slot_width = max(10, int((region_width - (gap_x * (cols - 1))) / cols))
        slot_height = max(10, int((region_height - (gap_y * (rows - 1))) / rows))
        slot_boxes: list[tuple[int, int, int, int]] = []
        for row_index in range(rows):
            slot_y1 = region_y1 + row_index * (slot_height + gap_y)
            slot_y2 = min(height, slot_y1 + slot_height)
            for col_index in range(cols):
                slot_x1 = region_x1 + col_index * (slot_width + gap_x)
                slot_x2 = min(width, slot_x1 + slot_width)
                slot_boxes.append((int(slot_x1), int(slot_y1), int(slot_x2), int(slot_y2)))
        template_meta = {
            "columns": cols,
            "rows": rows,
            "support_row_count": 0,
            "region_bbox": {"x1": int(region_x1), "y1": int(region_y1), "x2": int(region_x2), "y2": int(region_y2)},
            "slot_width": int(slot_width),
            "slot_height": int(slot_height),
            "gap_x": int(gap_x),
            "gap_y": int(gap_y),
            "source": "grid_fallback",
        }
        return slot_boxes, template_meta

    support_x1 = min(box[0] for box in support_boxes)
    support_x2 = max(box[2] for box in support_boxes)
    support_heights = [max(1, box[3] - box[1]) for box in support_boxes]
    median_support_height = float(median(support_heights))

    # Build deterministic column spans using clustered support-box centers.
    center_x_values = [((box[0] + box[2]) / 2.0) for box in support_boxes]
    column_centers = _kmeans_1d(center_x_values, cluster_count=cols)
    if len(column_centers) < cols:
        step = max(20.0, float(width - (2 * default_margin_x)) / cols)
        column_centers = [default_margin_x + (step * (index + 0.5)) for index in range(cols)]
    column_centers = sorted(column_centers[:cols])

    columns: list[list[tuple[int, int, int, int]]] = [[] for _ in range(cols)]
    for box in support_boxes:
        center_x = (box[0] + box[2]) / 2.0
        nearest_col = min(range(cols), key=lambda idx: abs(center_x - column_centers[idx]))
        columns[nearest_col].append(box)

    column_spans: list[tuple[int, int]] = []
    support_span_width = max(10, support_x2 - support_x1)
    fallback_column_width = max(10, int(round(support_span_width / cols)))
    for col_index in range(cols):
        column_boxes = columns[col_index]
        if column_boxes:
            column_x1 = int(round(median([box[0] for box in column_boxes])))
            column_x2 = int(round(median([box[2] for box in column_boxes])))
        else:
            column_x1 = support_x1 + (col_index * fallback_column_width)
            column_x2 = column_x1 + fallback_column_width
        column_x1 = max(0, min(width - 1, column_x1))
        column_x2 = max(column_x1 + 10, min(width, column_x2))
        column_spans.append((column_x1, column_x2))
    column_spans.sort(key=lambda span: span[0])

    # Derive row starts from support boxes. If OCR/contours miss the top row,
    # extrapolate one row upward using the observed row pitch.
    row_groups = _group_numeric_values(
        [float(box[1]) for box in support_boxes],
        threshold=max(6.0, median_support_height * 0.35),
    )
    row_starts = sorted({int(round(median(group))) for group in row_groups if group})
    support_row_count = len(row_starts)

    if len(row_starts) >= 2:
        row_diffs = [row_starts[index + 1] - row_starts[index] for index in range(len(row_starts) - 1)]
        row_pitch = int(round(median(row_diffs)))
    else:
        row_pitch = int(round(median_support_height + max(4.0, median_support_height * 0.04)))
    row_pitch = max(20, row_pitch)

    target_rows = max(default_rows, support_row_count)
    if not row_starts:
        row_starts = [default_top + (index * row_pitch) for index in range(target_rows)]
    elif len(row_starts) < target_rows:
        missing_rows = target_rows - len(row_starts)
        while missing_rows > 0 and row_starts and (row_starts[0] - row_pitch) >= int(height * 0.03):
            row_starts.insert(0, row_starts[0] - row_pitch)
            missing_rows -= 1
        while missing_rows > 0:
            row_starts.append(row_starts[-1] + row_pitch)
            missing_rows -= 1
    elif len(row_starts) > target_rows:
        row_starts = row_starts[:target_rows]

    row_height = int(round(median_support_height))
    if len(row_starts) >= 2:
        row_height = min(row_height, max(10, row_pitch - 4))
    row_height = max(10, row_height)

    slot_boxes: list[tuple[int, int, int, int]] = []
    for row_index, slot_y1_raw in enumerate(row_starts):
        slot_y1 = max(0, int(slot_y1_raw))
        slot_y2 = min(height, int(slot_y1 + row_height))
        if slot_y2 <= slot_y1:
            continue
        for column_x1, column_x2 in column_spans:
            slot_x1 = max(0, int(column_x1))
            slot_x2 = min(width, int(column_x2))
            if slot_x2 <= slot_x1:
                continue
            slot_boxes.append((slot_x1, slot_y1, slot_x2, slot_y2))

    if not slot_boxes:
        return _build_grid_card_boxes(width, height), {
            "columns": cols,
            "rows": default_rows,
            "support_row_count": support_row_count,
            "region_bbox": {"x1": int(default_margin_x), "y1": int(default_top), "x2": int(width - default_margin_x), "y2": int(height - default_bottom)},
            "slot_width": int(round((width - (2 * default_margin_x)) / cols)),
            "slot_height": int(round((height - default_top - default_bottom) / max(default_rows, 1))),
            "gap_x": 0,
            "gap_y": 0,
            "source": "grid_fallback_empty_template",
        }

    slot_boxes.sort(key=lambda box: (box[1], box[0]))
    region_x1 = min(box[0] for box in slot_boxes)
    region_y1 = min(box[1] for box in slot_boxes)
    region_x2 = max(box[2] for box in slot_boxes)
    region_y2 = max(box[3] for box in slot_boxes)
    template_meta = {
        "columns": cols,
        "rows": int(len(slot_boxes) / cols) if cols > 0 else 0,
        "support_row_count": support_row_count,
        "region_bbox": {"x1": int(region_x1), "y1": int(region_y1), "x2": int(region_x2), "y2": int(region_y2)},
        "slot_width": int(round(median([box[2] - box[0] for box in slot_boxes]))),
        "slot_height": int(round(median([box[3] - box[1] for box in slot_boxes]))),
        "gap_x": int(round(median([column_spans[index + 1][0] - column_spans[index][1] for index in range(len(column_spans) - 1)]))) if len(column_spans) > 1 else 0,
        "gap_y": int(row_pitch),
        "source": "support_aligned_template",
    }
    return slot_boxes, template_meta


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


def _ocr_card_image(
    card_image,
    timeout_seconds: float | None,
    *,
    fast_mode: bool = False,
) -> tuple[str, str | None, dict[str, Any]]:
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
        if fast_mode:
            ocr_meta["attempts"] = [
                {"source": "preprocessed", "config": "psm6", "score": _ocr_text_score(processed_text)},
            ]
            text = processed_text
        else:
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


def _ocr_digits_from_image(card_image, timeout_seconds: float | None) -> tuple[str, str | None, str]:
    try:
        import pytesseract
    except Exception as exc:  # noqa: BLE001
        return "", f"pytesseract import failed: {exc}", "none"

    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return "", "tesseract command not found", "none"

    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except Exception:  # noqa: BLE001
        pass

    preprocessed_image, preprocess_note = _preprocess_card_for_ocr(card_image)
    digit_configs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789",
    ]
    digit_errors: list[str] = []
    for config in digit_configs:
        try:
            digit_text = _run_tesseract(preprocessed_image, timeout_seconds=timeout_seconds, config=config)
            digits_only = re.sub(r"\D", "", str(digit_text or ""))
            if digits_only:
                return digits_only, None, preprocess_note
        except RuntimeError as exc:
            digit_errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            digit_errors.append(str(exc))
    if digit_errors:
        return "", f"digit-ocr failed: {digit_errors[0]}", preprocess_note
    return "", None, preprocess_note


def _ocr_id_from_image(card_image, timeout_seconds: float | None) -> tuple[str, str | None, str]:
    try:
        import pytesseract
    except Exception as exc:  # noqa: BLE001
        return "", f"pytesseract import failed: {exc}", "none"

    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return "", "tesseract command not found", "none"

    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except Exception:  # noqa: BLE001
        pass

    preprocessed_image, preprocess_note = _preprocess_card_for_ocr(card_image)
    id_configs = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/",
    ]
    id_errors: list[str] = []
    for config in id_configs:
        try:
            id_text = _run_tesseract(preprocessed_image, timeout_seconds=timeout_seconds, config=config)
            cleaned = re.sub(r"[^A-Za-z0-9/\\-]+", "", str(id_text or "").upper())
            if cleaned:
                return cleaned, None, preprocess_note
        except RuntimeError as exc:
            id_errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            id_errors.append(str(exc))
    if id_errors:
        return "", f"id-ocr failed: {id_errors[0]}", preprocess_note
    return "", None, preprocess_note


def _split_card_zones(card_image):
    width, height = card_image.size
    if width <= 2 or height <= 2:
        return {
            "serial": card_image,
            "elector_id": card_image,
            "body": card_image,
            "zone_bboxes": {
                "serial": (0, 0, width, height),
                "elector_id": (0, 0, width, height),
                "body": (0, 0, width, height),
            },
        }

    top_height = max(10, int(round(height * 0.30)))
    serial_split_x = max(8, int(round(width * 0.44)))
    body_start_y = max(1, int(round(height * 0.16)))
    # Exclude the right-side photo area from body OCR for this fixed electoral-roll card format.
    body_end_x = max(int(width * 0.60), int(round(width * 0.78)))

    serial_bbox = (0, 0, serial_split_x, top_height)
    elector_bbox = (max(0, serial_split_x - int(width * 0.03)), 0, width, top_height)
    body_bbox = (0, body_start_y, min(width, body_end_x), height)

    return {
        "serial": card_image.crop(serial_bbox),
        "elector_id": card_image.crop(elector_bbox),
        "body": card_image.crop(body_bbox),
        "zone_bboxes": {
            "serial": serial_bbox,
            "elector_id": elector_bbox,
            "body": body_bbox,
        },
    }


def _ocr_card_zones(
    card_image,
    timeout_seconds: float | None,
    *,
    use_expensive_passes: bool = True,
) -> dict[str, Any]:
    zones_payload = _split_card_zones(card_image)
    serial_text = ""
    serial_error = None
    serial_meta: dict[str, Any] = {}
    serial_digits_text, serial_digits_error, serial_digits_preprocess = _ocr_digits_from_image(
        zones_payload["serial"],
        timeout_seconds=timeout_seconds,
    )
    elector_text, elector_error, elector_verify_preprocess = _ocr_id_from_image(
        zones_payload["elector_id"],
        timeout_seconds=timeout_seconds,
    )
    elector_meta: dict[str, Any] = {"preprocess": elector_verify_preprocess}
    card_width, card_height = card_image.size
    top_micro_height = max(10, int(round(card_height * 0.22)))
    serial_micro_bbox = (
        max(0, int(round(card_width * 0.18))),
        0,
        min(card_width, int(round(card_width * 0.52))),
        min(card_height, top_micro_height),
    )
    elector_micro_bbox = (
        max(0, int(round(card_width * 0.45))),
        0,
        card_width,
        min(card_height, top_micro_height),
    )
    serial_verify_text = str(serial_digits_text or "").strip()
    serial_verify_error = serial_digits_error
    serial_verify_preprocess = serial_digits_preprocess
    elector_verify_text = str(elector_text or "").strip()
    elector_verify_error = elector_error
    body_text, body_error, body_meta = _ocr_card_image(
        zones_payload["body"],
        timeout_seconds=timeout_seconds,
        fast_mode=not use_expensive_passes,
    )

    if use_expensive_passes:
        serial_text, serial_error, serial_meta = _ocr_card_image(
            zones_payload["serial"],
            timeout_seconds=timeout_seconds,
        )
        serial_micro_image = card_image.crop(serial_micro_bbox)
        elector_micro_image = card_image.crop(elector_micro_bbox)
        serial_verify_text, serial_verify_error, serial_verify_preprocess = _ocr_digits_from_image(
            serial_micro_image,
            timeout_seconds=timeout_seconds,
        )
        elector_verify_text, elector_verify_error, elector_verify_preprocess = _ocr_id_from_image(
            elector_micro_image,
            timeout_seconds=timeout_seconds,
        )
        elector_zone_text, elector_zone_error, elector_zone_meta = _ocr_card_image(
            zones_payload["elector_id"],
            timeout_seconds=timeout_seconds,
        )
        if elector_zone_text:
            elector_text = elector_zone_text
        if elector_zone_error and not elector_error:
            elector_error = elector_zone_error
        if elector_zone_meta:
            elector_meta = elector_zone_meta
    else:
        # Fast path: use focused OCR outputs directly for top zones.
        serial_text = str(serial_digits_text or "").strip()

    combined_text = "\n".join(
        [
            str(serial_text or "").strip(),
            str(elector_text or "").strip(),
            str(body_text or "").strip(),
        ]
    ).strip()
    combined_errors = [error for error in [serial_error, elector_error, body_error] if error]

    return {
        "combined_text": combined_text,
        "combined_error": "; ".join(combined_errors) if combined_errors else None,
        "serial_text": str(serial_text or "").strip(),
        "serial_digits_text": str(serial_digits_text or "").strip(),
        "serial_verify_text": str(serial_verify_text or "").strip(),
        "elector_text": str(elector_text or "").strip(),
        "elector_verify_text": str(elector_verify_text or "").strip(),
        "body_text": str(body_text or "").strip(),
        "serial_error": serial_error,
        "serial_digits_error": serial_digits_error,
        "serial_verify_error": serial_verify_error,
        "elector_error": elector_error,
        "elector_verify_error": elector_verify_error,
        "body_error": body_error,
        "zone_bboxes": {
            **(zones_payload.get("zone_bboxes") or {}),
            "serial_micro": serial_micro_bbox,
            "elector_micro": elector_micro_bbox,
        },
        "preprocess": {
            "serial": (serial_meta or {}).get("preprocess"),
            "serial_digits": serial_digits_preprocess,
            "serial_verify": serial_verify_preprocess,
            "elector_id": (elector_meta or {}).get("preprocess"),
            "elector_verify": elector_verify_preprocess,
            "body": (body_meta or {}).get("preprocess"),
        },
    }


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
    cleaned = cleaned.replace("[", " ").replace("]", " ")
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :|-.,;")
    cleaned = re.sub(r"[^A-Za-z0-9/\-.' ]+$", "", cleaned).strip(" :|-.,;")
    if _is_junk_field_value(cleaned):
        return ""
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


def _normalize_card_label_variants(cleaned_text: str) -> tuple[str, list[str]]:
    normalized_text = str(cleaned_text or "")
    if not normalized_text:
        return "", []

    replacements: list[tuple[str, str, str]] = [
        (r"(?im)\bfath(?:er|ore|ere|or|ar|ers?)\s+(?:name|narne|nane)\b", "Fathers Name", "Fathers Name"),
        (r"(?im)\bhusb(?:and|ands|end|ends|an|ens?)\s+(?:name|narne|nane)\b", "Husbands Name", "Husbands Name"),
        (r"(?im)\bmoth(?:er|ers?|ore|ere)\s+(?:name|narne|nane)\b", "Mothers Name", "Mothers Name"),
        (r"(?im)\b(?:ags|aqe|agc|ago|ag)\b", "Age", "Age"),
        (r"(?im)\b(?:gonder|gendor|gendor|gendar)\b", "Gender", "Gender"),
    ]
    normalized_labels: set[str] = set()
    for pattern, canonical_text, canonical_label in replacements:
        if re.search(pattern, normalized_text):
            normalized_text = re.sub(pattern, canonical_text, normalized_text)
            normalized_labels.add(canonical_label)
    return normalized_text, sorted(normalized_labels)


def _normalize_gender_token(raw_gender: str) -> str:
    token = re.sub(r"[^a-z]", "", str(raw_gender or "").lower())
    if not token:
        return ""
    if token in {"m", "male"} or token.startswith("mal"):
        return "male"
    if token in {"f", "female"} or token.startswith("fem"):
        return "female"
    if token.startswith("oth"):
        return "other"
    return ""


def _is_valid_elector_id(value: str) -> bool:
    normalized = str(value or "").strip().upper()
    if not normalized:
        return False
    return bool(
        re.fullmatch(r"[A-Z]{3}\d{6,10}", normalized)
        or re.fullmatch(r"[A-Z]{2}/\d{2,3}/\d{2,3}/\d{4,8}", normalized)
    )


def _elector_id_format(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if re.fullmatch(r"[A-Z]{3}\d{6,10}", normalized):
        return "compact"
    if re.fullmatch(r"[A-Z]{2}/\d{2,3}/\d{2,3}/\d{4,8}", normalized):
        return "slash"
    return "invalid"


def _is_junk_field_value(value: str) -> bool:
    normalized = str(value or "").strip().strip(" .:-|").lower()
    if not normalized:
        return True
    if normalized in {
        "available",
        "photo",
        "number",
        "null",
        "none",
        "na",
        "n/a",
        "nil",
    }:
        return True
    if re.fullmatch(r"v\d+", normalized):
        return True
    return False


def _extract_labeled_value_strict(cleaned_text: str, label_patterns: Sequence[str]) -> str:
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        for label_pattern in label_patterns:
            match = re.search(
                rf"(?i)^\s*{label_pattern}\b\s*(?:[:\-|]\s*|\s+)(.*)$",
                line,
            )
            if not match:
                continue
            immediate_value = _clean_card_field(match.group(1))
            if immediate_value and not _is_junk_field_value(immediate_value):
                return immediate_value
            if index + 1 < len(lines):
                next_line = lines[index + 1]
                if ":" in next_line:
                    continue
                next_value = _clean_card_field(next_line)
                if next_value and not _is_junk_field_value(next_value):
                    return next_value
    return ""


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


def _parse_serial_number(cleaned_text: str, *, allow_loose: bool = False) -> str:
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

    for line in lines:
        if re.fullmatch(r"\d{1,4}", line):
            return line
    if allow_loose and lines:
        first_line_match = re.match(r"^(\d{1,5})\b", lines[0])
        if first_line_match:
            return first_line_match.group(1)
    return ""


def _parse_elector_id(cleaned_text: str) -> str:
    digit_to_letter = str.maketrans(
        {
            "0": "O",
            "1": "I",
            "5": "S",
            "6": "G",
            "8": "B",
        }
    )
    letter_to_digit = str.maketrans(
        {
            "O": "0",
            "Q": "0",
            "I": "1",
            "L": "1",
            "S": "5",
            "B": "8",
        }
    )

    def normalize_candidate(raw_candidate: str) -> str:
        candidate = str(raw_candidate or "").upper()
        candidate = candidate.replace("\\", "/").replace("|", "/")
        candidate = re.sub(r"\s+", "", candidate)
        candidate = re.sub(r"/{2,}", "/", candidate)
        candidate = re.sub(r"-{2,}", "-", candidate)

        if "/" in candidate:
            parts = [segment for segment in candidate.split("/") if segment]
            if len(parts) < 4:
                return ""
            prefix = re.sub(r"[^A-Z0-9]", "", parts[0]).translate(digit_to_letter)
            if prefix == "W":
                prefix = "WB"
            if prefix.startswith("WB"):
                prefix = "WB"
            if len(prefix) != 2:
                prefix = prefix[:2]
            if len(prefix) != 2 or not prefix.isalpha():
                return ""

            numeric_parts: list[str] = []
            for segment in parts[1:4]:
                normalized_segment = re.sub(r"[^A-Z0-9]", "", segment).translate(letter_to_digit)
                normalized_segment = re.sub(r"\D", "", normalized_segment)
                if not normalized_segment:
                    return ""
                numeric_parts.append(normalized_segment)
            return "/".join([prefix, numeric_parts[0], numeric_parts[1], numeric_parts[2]])

        compact_candidate = re.sub(r"[^A-Z0-9]", "", candidate)
        if not compact_candidate:
            return ""

        # Prefer classic EPIC form (3-letter prefix + digits), but tolerate OCR drift.
        def normalize_compact_parts(prefix_raw: str, suffix_raw: str) -> str:
            prefix = str(prefix_raw or "").translate(digit_to_letter)
            suffix = str(suffix_raw or "")
            if len(prefix) == 2 and suffix and suffix[0] in "OILSB":
                # Recover dropped 3rd prefix char, e.g. "AR" + "O2700045" => "ARO2700045".
                prefix = (prefix + suffix[0]).translate(digit_to_letter)
                suffix = suffix[1:]
            if len(prefix) == 4 and prefix.endswith(("I", "L")) and prefix[:3].isalpha():
                prefix = prefix[:3]
            if prefix.startswith("BGNI"):
                prefix = "BGN"
            if len(prefix) > 3:
                prefix = "BGN" if prefix.startswith("BGN") else prefix[:3]
            if len(prefix) != 3 or not prefix.isalpha():
                return ""
            numeric_suffix = suffix.translate(letter_to_digit)
            numeric_suffix = re.sub(r"\D", "", numeric_suffix)
            if len(numeric_suffix) < 6 or len(numeric_suffix) > 10:
                return ""
            return prefix + numeric_suffix

        compact_patterns = [
            r"([A-Z0-9]{3})([0-9OILSB]{6,10})",
            r"([A-Z0-9]{4})([0-9OILSB]{6,10})",
            r"([A-Z0-9]{2})([0-9OILSB]{7,11})",
        ]
        for compact_pattern in compact_patterns:
            for compact_match in re.finditer(compact_pattern, compact_candidate):
                normalized_compact = normalize_compact_parts(
                    compact_match.group(1),
                    compact_match.group(2),
                )
                if _is_valid_elector_id(normalized_compact):
                    return normalized_compact
        return ""

    labeled_id = _extract_labeled_value(
        cleaned_text,
        [r"epic", r"elector\s*id", r"voter\s*id", r"id\s*no\.?", r"card\s*no\.?"],
    )
    if labeled_id:
        normalized_labeled = normalize_candidate(labeled_id)
        if _is_valid_elector_id(normalized_labeled):
            return normalized_labeled

    upper_text = cleaned_text.upper()
    patterns = [
        r"([A-Z0-9]{2,5}\s*(?:[0-9OILSB]\s*){6,10})",
        r"([A-Z0-9]{1,3}\s*/\s*[A-Z0-9]{2,3}\s*/\s*[A-Z0-9]{2,3}\s*/\s*[A-Z0-9]{4,8})",
        r"([A-Z0-9]{1,3}\s*[-/]\s*[A-Z0-9]{6,10})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, upper_text):
            candidate = match.group(1)
            normalized = normalize_candidate(candidate)
            if _is_valid_elector_id(normalized):
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


def _missing_fields_for_record(record: dict[str, Any]) -> list[str]:
    checks = [
        ("serial_number", str(record.get("serial_number") or "").strip()),
        ("elector_id", str(record.get("elector_id") or "").strip()),
        ("name", str(record.get("name") or "").strip()),
        ("relative_type", str(record.get("relative_type") or "").strip()),
        ("relative_name", str(record.get("relative_name") or "").strip()),
        ("house_number", str(record.get("house_number") or "").strip()),
        ("age", record.get("age")),
        ("gender", str(record.get("gender") or "").strip()),
    ]
    missing: list[str] = []
    for field_name, value in checks:
        if value is None:
            missing.append(field_name)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field_name)
    return missing


def _parse_status_rank(parse_status: str) -> int:
    if parse_status == "valid":
        return 3
    if parse_status == "partial_field_missing":
        return 2
    if parse_status == "partial_top_missing":
        return 1
    return 0


def _classify_card_record(record: dict[str, Any], *, body_has_lower_signals: bool) -> tuple[str, str]:
    name_value = str(record.get("name") or "").strip()
    if not name_value:
        if body_has_lower_signals:
            return "partial_top_missing", "name missing but lower fields present"
        return "rejected_noise", "name missing"
    if _is_noise_name(name_value):
        return "rejected_noise", "invalid/noisy name"

    elector_id_value = str(record.get("elector_id") or "").strip()
    if elector_id_value and not _is_valid_elector_id(elector_id_value):
        return "rejected_noise", "invalid elector_id format"

    serial_number_value = str(record.get("serial_number") or "").strip()
    if serial_number_value and not re.fullmatch(r"\d{1,5}", serial_number_value):
        return "rejected_noise", "invalid serial_number format"

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

    if primary_fields_present >= 3:
        return "valid", ""
    if primary_fields_present >= 2 and (elector_id_value or serial_number_value):
        return "valid", ""
    if not elector_id_value and not serial_number_value and primary_fields_present >= 1:
        return "partial_top_missing", "serial/elector missing"
    if primary_fields_present >= 1:
        return "partial_field_missing", "missing some key fields"
    if secondary_fields_present >= 1:
        if body_has_lower_signals:
            return "partial_top_missing", "top fields missing"
        return "partial_field_missing", "only secondary fields parsed"
    return "rejected_noise", "missing minimum identity fields"


def _classify_serial_confidence(
    *,
    serial_number: str,
    parse_quality: str,
    slot_index: int | None,
    expected_slots: int | None,
    serial_pass_a_text: str,
    serial_pass_b_text: str,
) -> tuple[str, str]:
    normalized = str(serial_number or "").strip()
    if not normalized:
        return "needs_review", "serial missing"
    if not re.fullmatch(r"\d{1,5}", normalized):
        return "needs_review", "serial invalid format"

    serial_int = int(normalized)
    if serial_int <= 0:
        return "needs_review", "serial must be positive"

    pass_a = re.sub(r"\D", "", str(serial_pass_a_text or "")).lstrip("0")
    pass_b = re.sub(r"\D", "", str(serial_pass_b_text or "")).lstrip("0")
    value_no_zero = normalized.lstrip("0") or normalized

    dual_pass_agree = bool(pass_a and pass_b and pass_a == pass_b == value_no_zero)
    any_pass_match = bool((pass_a and pass_a == value_no_zero) or (pass_b and pass_b == value_no_zero))

    local_range_ok = True
    if expected_slots is not None and int(expected_slots) > 0:
        max_expected = max(int(expected_slots) + 5, int(expected_slots) * 2)
        local_range_ok = 1 <= serial_int <= max_expected
    if not local_range_ok:
        return "needs_review", "serial out of expected local range"

    order_close = True
    slot_exact = False
    if slot_index is not None and int(slot_index) > 0:
        slot_exact = serial_int == int(slot_index)
        order_close = abs(serial_int - int(slot_index)) <= 1

    strong_quality = parse_quality in {
        "direct_zone_match",
        "direct_strong_match",
        "normalized_fuzzy_match",
    }

    if parse_quality in {"direct_strong_match", "direct_zone_match"} and (dual_pass_agree or (any_pass_match and order_close)):
        return "trusted", "serial verified by zone/micro pass"
    if dual_pass_agree and order_close:
        return "trusted", "serial verified by dual-pass agreement"
    if slot_exact and strong_quality:
        return "trusted", "serial aligned with slot order"
    if any_pass_match and order_close and strong_quality:
        return "trusted", "serial matched in one focused OCR pass"
    if any_pass_match or order_close:
        return "low_confidence", "serial weak agreement"
    return "needs_review", "serial mismatch across passes"


def _classify_elector_confidence(
    *,
    elector_id: str,
    parse_quality: str,
    elector_pass_a_text: str,
    elector_pass_b_text: str,
) -> tuple[str, str]:
    normalized = str(elector_id or "").strip().upper()
    if not normalized:
        return "needs_review", "elector_id missing"
    if not _is_valid_elector_id(normalized):
        return "needs_review", "elector_id invalid format"

    pass_a = _parse_elector_id(_clean_card_ocr_text(elector_pass_a_text))
    pass_b = _parse_elector_id(_clean_card_ocr_text(elector_pass_b_text))
    dual_pass_agree = bool(pass_a and pass_b and pass_a == pass_b == normalized)
    any_pass_match = bool((pass_a and pass_a == normalized) or (pass_b and pass_b == normalized))

    if dual_pass_agree:
        return "trusted", "elector_id verified by dual-pass agreement"
    if parse_quality == "direct_zone_match" and any_pass_match:
        return "trusted", "elector_id verified in zone pass"
    if any_pass_match:
        return "low_confidence", "elector_id verified in one pass"
    if pass_a and pass_b and pass_a != pass_b:
        return "needs_review", "elector_id pass disagreement"
    return "low_confidence", "elector_id single-source parse"


def _parse_card_record(
    *,
    raw_card_text: str,
    cleaned_card_text: str,
    serial_zone_text: str,
    serial_zone_digits_text: str,
    serial_zone_verify_text: str,
    elector_zone_text: str,
    elector_zone_verify_text: str,
    body_zone_text: str,
    file_name: str,
    file_path: str,
    page_number: int,
    constituency: str | None,
    section_name: str | None,
    extraction_method: str,
    slot_index: int | None = None,
    expected_slots: int | None = None,
) -> tuple[dict[str, Any] | None, str, str | None, list[str], dict[str, Any]]:
    cleaned_serial_zone = _clean_card_ocr_text(serial_zone_text)
    cleaned_serial_verify_zone = _clean_card_ocr_text(serial_zone_verify_text)
    cleaned_elector_zone = _clean_card_ocr_text(elector_zone_text)
    cleaned_elector_verify_zone = _clean_card_ocr_text(elector_zone_verify_text)
    cleaned_body_zone = _clean_card_ocr_text(body_zone_text)
    normalized_labels_detected: set[str] = set()
    cleaned_serial_zone, serial_labels = _normalize_card_label_variants(cleaned_serial_zone)
    cleaned_serial_verify_zone, serial_verify_labels = _normalize_card_label_variants(cleaned_serial_verify_zone)
    cleaned_elector_zone, elector_labels = _normalize_card_label_variants(cleaned_elector_zone)
    cleaned_elector_verify_zone, elector_verify_labels = _normalize_card_label_variants(cleaned_elector_verify_zone)
    cleaned_body_zone, body_labels = _normalize_card_label_variants(cleaned_body_zone)
    normalized_labels_detected.update(serial_labels)
    normalized_labels_detected.update(serial_verify_labels)
    normalized_labels_detected.update(elector_labels)
    normalized_labels_detected.update(elector_verify_labels)
    normalized_labels_detected.update(body_labels)

    parse_meta: dict[str, Any] = {
        "field_parse_quality": {},
        "normalized_labels_detected": sorted(normalized_labels_detected),
        "serial_candidates": [],
        "elector_candidates": [],
        "elector_raw_candidates": [],
        "serial_ocr_pass_a_raw": str(serial_zone_text or ""),
        "serial_ocr_pass_b_raw": str(serial_zone_verify_text or ""),
        "elector_ocr_pass_a_raw": str(elector_zone_text or ""),
        "elector_ocr_pass_b_raw": str(elector_zone_verify_text or ""),
        "serial_number_cleaned": None,
        "elector_id_cleaned": None,
        "elector_id_format": "",
        "serial_confidence": "needs_review",
        "elector_confidence": "needs_review",
        "record_status": "needs_review",
        "record_status_reason": "",
    }
    if not cleaned_card_text.strip():
        cleaned_card_text = "\n".join(
            [segment for segment in [cleaned_serial_zone, cleaned_elector_zone, cleaned_body_zone] if segment]
        ).strip()
    cleaned_card_text, card_labels = _normalize_card_label_variants(cleaned_card_text)
    normalized_labels_detected.update(card_labels)
    parse_meta["normalized_labels_detected"] = sorted(normalized_labels_detected)
    if not cleaned_card_text.strip():
        parse_meta["field_parse_quality"] = {
            "serial_number": "missing",
            "elector_id": "missing",
            "name": "missing",
        }
        return None, "rejected_noise", "empty card text after cleanup", [
            "serial_number",
            "elector_id",
            "name",
        ], parse_meta

    parse_text = cleaned_body_zone or cleaned_card_text
    field_quality: dict[str, str] = {}

    serial_number = ""
    serial_quality = "missing"
    serial_candidates: list[tuple[str, str]] = []
    serial_digits_candidate = re.sub(r"\D", "", str(serial_zone_digits_text or ""))
    if serial_digits_candidate and len(serial_digits_candidate) <= 4:
        serial_candidates.append((serial_digits_candidate.lstrip("0") or serial_digits_candidate, "zone_digits_pass_a"))
    serial_verify_candidate = re.sub(r"\D", "", str(serial_zone_verify_text or ""))
    if serial_verify_candidate and len(serial_verify_candidate) <= 4:
        serial_candidates.append((serial_verify_candidate.lstrip("0") or serial_verify_candidate, "micro_digits_pass_b"))
    for candidate_value, candidate_source in serial_candidates:
        if re.fullmatch(r"\d{1,5}", candidate_value):
            serial_number = candidate_value
            serial_quality = "direct_strong_match"
            if candidate_source == "zone_digits_pass_a":
                break
    if not serial_number:
        serial_zone_candidate = _parse_serial_number(cleaned_serial_zone)
        if serial_zone_candidate and re.fullmatch(r"\d{1,5}", serial_zone_candidate):
            serial_number = serial_zone_candidate
            serial_quality = "direct_zone_match"
    if not serial_number:
        body_serial = _parse_serial_number(cleaned_body_zone, allow_loose=True)
        if body_serial and re.fullmatch(r"\d{1,5}", body_serial):
            serial_number = body_serial
            serial_quality = "fallback_inference"
    if not serial_number:
        fallback_serial = _parse_serial_number(cleaned_card_text, allow_loose=True)
        if fallback_serial and re.fullmatch(r"\d{1,5}", fallback_serial):
            serial_number = fallback_serial
            serial_quality = "fallback_inference"
    if serial_number and not re.fullmatch(r"\d{1,5}", serial_number):
        serial_number = ""
        serial_quality = "invalid_rejected"
    parse_meta["serial_number_cleaned"] = serial_number or None
    parse_meta["serial_candidates"] = [candidate for candidate, _ in serial_candidates] + [
        value
        for value in [serial_number]
        if value
    ]
    field_quality["serial_number"] = serial_quality if serial_number else "missing"

    elector_id = ""
    elector_quality = "missing"
    elector_candidates: list[str] = []
    elector_raw_candidates: list[dict[str, str]] = []

    def _consume_elector_source(source_name: str, source_text: str, quality_name: str) -> None:
        nonlocal elector_id, elector_quality
        raw_value = str(source_text or "")
        if raw_value.strip():
            elector_raw_candidates.append({"source": source_name, "raw": raw_value[:240]})
        candidate_value = _parse_elector_id(raw_value)
        if candidate_value:
            if candidate_value not in elector_candidates:
                elector_candidates.append(candidate_value)
            elector_raw_candidates.append(
                {
                    "source": source_name,
                    "raw": raw_value[:240],
                    "normalized": candidate_value,
                    "format": _elector_id_format(candidate_value),
                }
            )
            if not elector_id:
                elector_id = candidate_value
                elector_quality = quality_name

    _consume_elector_source("zone_pass_a", cleaned_elector_zone, "direct_zone_match")
    _consume_elector_source("zone_pass_b", cleaned_elector_verify_zone, "direct_strong_match")
    _consume_elector_source("full_card", cleaned_card_text, "normalized_fuzzy_match")
    _consume_elector_source("body_zone", cleaned_body_zone, "fallback_inference")
    if elector_id and not _is_valid_elector_id(elector_id):
        elector_id = ""
        elector_quality = "invalid_rejected"
    parse_meta["elector_id_cleaned"] = elector_id or None
    parse_meta["elector_id_format"] = _elector_id_format(elector_id) if elector_id else ""
    parse_meta["elector_candidates"] = elector_candidates
    parse_meta["elector_raw_candidates"] = elector_raw_candidates
    field_quality["elector_id"] = elector_quality if elector_id else "missing"

    name = _extract_labeled_value_strict(
        parse_text,
        [
            r"name(?:\s+of\s+elector)?",
            r"elector'?s\s+name",
        ],
    )
    if not name:
        name = _extract_labeled_value_strict(
            cleaned_card_text,
            [
                r"name(?:\s+of\s+elector)?",
                r"elector'?s\s+name",
            ],
        )
    name = _clean_card_field(name)
    field_quality["name"] = "direct_strong_match" if name else "missing"

    relative_type = ""
    relative_name = ""
    strict_relative_map: list[tuple[str, list[str]]] = [
        ("father", [r"fathers?\s+name", r"father'?s\s+name"]),
        ("husband", [r"husbands?\s+name", r"husband'?s\s+name"]),
        ("mother", [r"mothers?\s+name", r"mother'?s\s+name"]),
    ]
    for rel_type, rel_labels in strict_relative_map:
        relative_candidate = _extract_labeled_value_strict(parse_text, rel_labels)
        if not relative_candidate:
            relative_candidate = _extract_labeled_value_strict(cleaned_card_text, rel_labels)
        if relative_candidate:
            relative_type = rel_type
            relative_name = _clean_card_field(relative_candidate)
            break
    field_quality["relative_name"] = "direct_strong_match" if relative_name else "missing"

    house_number = _extract_labeled_value_strict(
        parse_text,
        [r"house\s+number", r"house\s+no\.?", r"house\s+no", r"h\.?\s*no\.?"],
    )
    if not house_number:
        house_number = _extract_labeled_value_strict(
            cleaned_card_text,
            [r"house\s+number", r"house\s+no\.?", r"house\s+no", r"h\.?\s*no\.?"],
        )
    house_number = _clean_card_field(house_number)
    field_quality["house_number"] = "direct_strong_match" if house_number else "missing"

    age = None
    age_quality = "missing"
    age_value = _extract_labeled_value_strict(parse_text, [r"age"])
    if not age_value:
        age_value = _extract_labeled_value_strict(cleaned_card_text, [r"age"])
    age_match = re.search(r"\b(\d{1,3})\b", age_value)
    if not age_match:
        line_age_match = re.search(r"(?im)\bage\b[^0-9]{0,6}(\d{1,3})\b", parse_text or cleaned_card_text)
        if line_age_match:
            age_match = line_age_match
            age_quality = "normalized_fuzzy_match"
    if age_match:
        age = int(age_match.group(1))
        if age <= 0 or age > 120:
            age = None
        else:
            if age_quality == "missing":
                age_quality = "direct_strong_match"
    field_quality["age"] = age_quality if age is not None else "missing"

    gender = ""
    gender_quality = "missing"
    gender_value = _extract_labeled_value_strict(parse_text, [r"gender"])
    if not gender_value:
        gender_value = _extract_labeled_value_strict(cleaned_card_text, [r"gender"])
        if gender_value:
            gender_quality = "normalized_fuzzy_match"
    if not gender_value:
        combined_age_gender = re.search(
            r"(?im)\bage\b[^0-9]{0,6}\d{1,3}\b.{0,20}\b(?:gender|sex)\b[^A-Za-z0-9]{0,6}([A-Za-z]{1,12})",
            parse_text or cleaned_card_text,
        )
        if combined_age_gender:
            gender_value = combined_age_gender.group(1)
            gender_quality = "fallback_inference"
    if not gender_value:
        direct_gender_match = re.search(
            r"(?im)\b(ma[li1][eec]?|fem[a-z]{1,5}|male|female|m|f|other)\b",
            parse_text or cleaned_card_text,
        )
        if direct_gender_match:
            gender_value = direct_gender_match.group(1)
            gender_quality = "fallback_inference"
    gender = _normalize_gender_token(gender_value)
    if gender and gender_quality == "missing":
        gender_quality = "direct_strong_match"
    field_quality["gender"] = gender_quality if gender else "missing"

    if _is_junk_field_value(relative_name):
        relative_name = ""
        relative_type = ""
    if _is_junk_field_value(house_number):
        house_number = ""

    body_house = _extract_labeled_value_strict(parse_text, [r"house\s+number", r"house\s+no\.?", r"h\.?\s*no\.?"])
    body_age_text = _extract_labeled_value_strict(parse_text, [r"age"])
    body_gender_text = _extract_labeled_value_strict(parse_text, [r"gender"])
    body_age = int(re.search(r"\b(\d{1,3})\b", body_age_text).group(1)) if re.search(r"\b(\d{1,3})\b", body_age_text) else None
    body_gender = body_gender_text
    body_has_lower_signals = bool(body_house or body_age is not None or body_gender)

    serial_confidence, serial_confidence_reason = _classify_serial_confidence(
        serial_number=serial_number,
        parse_quality=field_quality.get("serial_number", "missing"),
        slot_index=slot_index,
        expected_slots=expected_slots,
        serial_pass_a_text=str(serial_zone_digits_text or serial_zone_text or ""),
        serial_pass_b_text=str(serial_zone_verify_text or ""),
    )
    elector_confidence, elector_confidence_reason = _classify_elector_confidence(
        elector_id=elector_id,
        parse_quality=field_quality.get("elector_id", "missing"),
        elector_pass_a_text=str(elector_zone_text or ""),
        elector_pass_b_text=str(elector_zone_verify_text or ""),
    )
    parse_meta["serial_confidence"] = serial_confidence
    parse_meta["serial_confidence_reason"] = serial_confidence_reason
    parse_meta["elector_confidence"] = elector_confidence
    parse_meta["elector_confidence_reason"] = elector_confidence_reason

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
    parse_meta["field_parse_quality"] = field_quality
    missing_fields = _missing_fields_for_record(parsed_record)
    parse_status, reason = _classify_card_record(parsed_record, body_has_lower_signals=body_has_lower_signals)

    if parse_status == "rejected_noise":
        record_status = "needs_review"
        record_status_reason = reason or "rejected"
    elif serial_confidence == "trusted" and elector_confidence == "trusted" and parse_status == "valid":
        record_status = "trusted"
        record_status_reason = "sensitive fields verified"
    elif serial_confidence == "needs_review" or elector_confidence == "needs_review":
        record_status = "needs_review"
        record_status_reason = "sensitive field invalid/weak"
    elif serial_confidence == "low_confidence" or elector_confidence == "low_confidence":
        record_status = "needs_review"
        record_status_reason = "sensitive field low confidence"
    elif parse_status != "valid":
        record_status = "partial"
        record_status_reason = parse_status
    else:
        record_status = "trusted"
        record_status_reason = "record validated"
    parse_meta["record_status"] = record_status
    parse_meta["record_status_reason"] = record_status_reason

    if parse_status == "rejected_noise":
        return None, parse_status, reason, missing_fields, parse_meta
    parsed_record["_parse_status"] = parse_status
    parsed_record["_record_status"] = record_status
    parsed_record["_record_status_reason"] = record_status_reason
    parsed_record["_missing_fields"] = missing_fields
    parsed_record["_field_parse_quality"] = dict(field_quality)
    parsed_record["_serial_confidence"] = serial_confidence
    parsed_record["_serial_confidence_reason"] = serial_confidence_reason
    parsed_record["_elector_confidence"] = elector_confidence
    parsed_record["_elector_confidence_reason"] = elector_confidence_reason
    parsed_record["_normalized_labels"] = sorted(normalized_labels_detected)
    return parsed_record, parse_status, reason or None, missing_fields, parse_meta


def _should_retry_top_capture(
    *,
    parse_status: str,
    missing_fields: Sequence[str],
    cleaned_body_text: str,
) -> bool:
    missing_set = set(missing_fields or [])
    top_field_missing = bool({"serial_number", "elector_id", "name"} & missing_set)
    if not top_field_missing:
        return False
    if parse_status == "partial_top_missing":
        return True
    body_house = _parse_house_number(cleaned_body_text)
    body_age, body_gender = _parse_age_and_gender(cleaned_body_text)
    return bool(body_house or body_age is not None or body_gender)


def _is_useful_record_for_insert(record: dict[str, Any] | None, parse_status: str) -> bool:
    if record is None:
        return False
    if parse_status == "valid":
        return True
    if parse_status not in {"partial_top_missing", "partial_field_missing"}:
        return False
    name_value = str(record.get("name") or "").strip()
    if not name_value or _is_noise_name(name_value):
        return False
    return any(
        [
            str(record.get("elector_id") or "").strip(),
            str(record.get("serial_number") or "").strip(),
            str(record.get("house_number") or "").strip(),
            record.get("age") is not None,
        ]
    )


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
        "slots_detected_support": 0,
        "slots_expected": 0,
        "slots_ocr_attempted": 0,
        "cards_with_text": 0,
        "cards_parsed": 0,
        "cards_valid": 0,
        "cards_partial": 0,
        "cards_rejected": 0,
        "cards_missing_top_fields": 0,
        "cards_top_retry_attempted": 0,
        "cards_top_retry_used": 0,
        "cards_inserted": 0,
        "expected_card_count": 0,
        "reject_reason_breakdown": {},
        "failed_slots": [],
        "failed_slot_indexes": [],
        "slot_template_meta": {},
        "header_bbox": {},
        "header_ocr_text": "",
        "header_ocr_error": None,
        "header_ocr_preprocess": None,
        "header_crop_png_bytes": None,
        "header_metadata": {},
        "metadata_values": {},
        "constituency_parsed": False,
        "section_name_parsed": False,
        "part_number_parsed": False,
        "records_metadata_propagated": 0,
        "records_total_for_insert": 0,
        "serial_number_filled_count": 0,
        "elector_id_valid_count": 0,
        "elector_id_compact_accepted_count": 0,
        "elector_id_slash_accepted_count": 0,
        "elector_id_trusted_count": 0,
        "gender_filled_count": 0,
        "records_partial_after_cleanup": 0,
        "trusted_records_count": 0,
        "needs_review_records_count": 0,
        "serial_low_confidence_count": 0,
        "elector_id_low_confidence_count": 0,
        "cards": [],
    }

    page_image, render_error = _render_pdf_page_image(file_path=file_path, page_number=page_number)
    if page_image is None:
        debug_payload["mode"] = "page_text_fallback"
        debug_payload["detection_error"] = render_error or "page image unavailable"
        debug_payload["cards_parsed"] = len(fallback_records)
        debug_payload["cards_valid"] = len(fallback_records)
        debug_payload["cards_partial"] = 0
        debug_payload["slots_expected"] = len(fallback_records)
        return {"records": fallback_records, "debug": debug_payload}

    support_boxes, detection_strategy, detection_error = _detect_voter_card_boxes(page_image)
    page_width, page_height = page_image.size
    slot_boxes, template_meta = _derive_template_slot_boxes(
        width=page_width,
        height=page_height,
        support_boxes=support_boxes,
    )
    debug_payload["detection_strategy"] = detection_strategy
    debug_payload["detection_error"] = detection_error
    debug_payload["cards_detected"] = len(support_boxes)
    debug_payload["slots_detected_support"] = len(support_boxes)
    debug_payload["slots_expected"] = len(slot_boxes)
    debug_payload["expected_card_count"] = len(slot_boxes)
    debug_payload["slot_template_meta"] = template_meta
    if not slot_boxes:
        debug_payload["mode"] = "page_text_fallback"
        debug_payload["cards_parsed"] = len(fallback_records)
        debug_payload["cards_valid"] = len(fallback_records)
        debug_payload["cards_partial"] = 0
        debug_payload["cards_rejected"] = 0
        return {"records": fallback_records, "debug": debug_payload}

    should_ocr_header = bool(include_card_debug) or not (str(constituency or "").strip() and str(section_name or "").strip())
    if should_ocr_header:
        header_payload = _extract_page_header_metadata(
            page_image=page_image,
            support_boxes=support_boxes,
            ocr_timeout_seconds=ocr_timeout_seconds,
            include_preview=bool(include_card_debug),
        )
    else:
        header_payload = {
            "bbox": {},
            "ocr_text": "",
            "ocr_error": None,
            "ocr_preprocess": None,
            "crop_png_bytes": None,
            "metadata": {
                "constituency": None,
                "section_name": None,
                "part_number": None,
            },
        }
    header_metadata = dict(header_payload.get("metadata") or {})
    header_constituency = str(header_metadata.get("constituency") or "").strip()
    header_section_name = str(header_metadata.get("section_name") or "").strip()
    header_part_number = str(header_metadata.get("part_number") or "").strip()
    page_text_part_number = _extract_part_number(page_text)

    # Prefer explicit page-header metadata, fall back to page-text context when needed.
    constituency = header_constituency or str(constituency or "").strip() or None
    section_name = header_section_name or str(section_name or "").strip() or None
    part_number = header_part_number or page_text_part_number or None

    debug_payload["header_bbox"] = dict(header_payload.get("bbox") or {})
    debug_payload["header_ocr_text"] = str(header_payload.get("ocr_text") or "")
    debug_payload["header_ocr_error"] = header_payload.get("ocr_error")
    debug_payload["header_ocr_preprocess"] = header_payload.get("ocr_preprocess")
    debug_payload["header_crop_png_bytes"] = header_payload.get("crop_png_bytes")
    debug_payload["header_metadata"] = {
        "constituency": constituency,
        "section_name": section_name,
        "part_number": part_number,
    }
    debug_payload["metadata_values"] = {
        "constituency": constituency,
        "section_name": section_name,
        "part_number": part_number,
    }
    debug_payload["constituency_parsed"] = bool(constituency)
    debug_payload["section_name_parsed"] = bool(section_name)
    debug_payload["part_number_parsed"] = bool(part_number)

    parsed_records: list[dict[str, Any]] = []
    cards_debug: list[dict[str, Any]] = []
    failed_slots: list[dict[str, Any]] = []
    retry_extra_top_ratio = _float_env("NAME_SEARCH_CARD_RETRY_TOP_EXTRA", 0.14, minimum=0.0, maximum=0.25)
    use_expensive_zone_ocr = bool(include_card_debug)

    def _confidence_score(confidence_value: str) -> int:
        normalized_value = str(confidence_value or "").strip()
        if normalized_value == "trusted":
            return 2
        if normalized_value == "low_confidence":
            return 1
        return 0

    for card_index, box in enumerate(slot_boxes, start=1):
        debug_payload["slots_ocr_attempted"] = int(debug_payload.get("slots_ocr_attempted", 0)) + 1
        original_x1, original_y1, original_x2, original_y2 = [int(value) for value in box]
        expanded_box = _expand_box(
            (original_x1, original_y1, original_x2, original_y2),
            width=page_width,
            height=page_height,
        )
        x1, y1, x2, y2 = expanded_box
        card_image = page_image.crop((x1, y1, x2, y2))

        zone_ocr_payload = _ocr_card_zones(
            card_image,
            timeout_seconds=ocr_timeout_seconds,
            use_expensive_passes=use_expensive_zone_ocr,
        )
        raw_card_text = str(zone_ocr_payload.get("combined_text") or "")
        ocr_error = zone_ocr_payload.get("combined_error")
        cleaned_card_text = _clean_card_ocr_text(raw_card_text)
        cleaned_body_text = _clean_card_ocr_text(str(zone_ocr_payload.get("body_text") or ""))
        if cleaned_card_text.strip():
            debug_payload["cards_with_text"] = int(debug_payload["cards_with_text"]) + 1

        record, parse_status, reject_reason, missing_fields, parse_meta = _parse_card_record(
            raw_card_text=raw_card_text,
            cleaned_card_text=cleaned_card_text,
            serial_zone_text=str(zone_ocr_payload.get("serial_text") or ""),
            serial_zone_digits_text=str(zone_ocr_payload.get("serial_digits_text") or ""),
            serial_zone_verify_text=str(zone_ocr_payload.get("serial_verify_text") or ""),
            elector_zone_text=str(zone_ocr_payload.get("elector_text") or ""),
            elector_zone_verify_text=str(zone_ocr_payload.get("elector_verify_text") or ""),
            body_zone_text=str(zone_ocr_payload.get("body_text") or ""),
            file_name=file_name,
            file_path=file_path,
            page_number=page_number,
            constituency=constituency,
            section_name=section_name,
            extraction_method=f"card_ocr_{detection_strategy}",
            slot_index=card_index,
            expected_slots=len(slot_boxes),
        )
        if not use_expensive_zone_ocr:
            initial_rank = _parse_status_rank(parse_status)
            initial_missing_count = len(missing_fields)
            initial_conf_score = _confidence_score(parse_meta.get("serial_confidence")) + _confidence_score(
                parse_meta.get("elector_confidence")
            )
            parsed_serial = str((record or {}).get("serial_number") or "").strip()
            parsed_elector = str((record or {}).get("elector_id") or "").strip()
            top_field_missing = bool({"serial_number", "elector_id", "name"} & set(missing_fields))
            sensitive_fields_weak = (
                parse_status in {"rejected_noise", "partial_top_missing", "partial_field_missing"}
                or top_field_missing
                or not parsed_serial
                or not parsed_elector
            )
            if sensitive_fields_weak:
                expensive_zone_ocr = _ocr_card_zones(
                    card_image,
                    timeout_seconds=ocr_timeout_seconds,
                    use_expensive_passes=True,
                )
                expensive_raw_text = str(expensive_zone_ocr.get("combined_text") or "")
                expensive_cleaned_text = _clean_card_ocr_text(expensive_raw_text)
                expensive_record, expensive_status, expensive_reason, expensive_missing_fields, expensive_parse_meta = _parse_card_record(
                    raw_card_text=expensive_raw_text,
                    cleaned_card_text=expensive_cleaned_text,
                    serial_zone_text=str(expensive_zone_ocr.get("serial_text") or ""),
                    serial_zone_digits_text=str(expensive_zone_ocr.get("serial_digits_text") or ""),
                    serial_zone_verify_text=str(expensive_zone_ocr.get("serial_verify_text") or ""),
                    elector_zone_text=str(expensive_zone_ocr.get("elector_text") or ""),
                    elector_zone_verify_text=str(expensive_zone_ocr.get("elector_verify_text") or ""),
                    body_zone_text=str(expensive_zone_ocr.get("body_text") or ""),
                    file_name=file_name,
                    file_path=file_path,
                    page_number=page_number,
                    constituency=constituency,
                    section_name=section_name,
                    extraction_method=f"card_ocr_{detection_strategy}",
                    slot_index=card_index,
                    expected_slots=len(slot_boxes),
                )
                expensive_rank = _parse_status_rank(expensive_status)
                expensive_conf_score = _confidence_score(expensive_parse_meta.get("serial_confidence")) + _confidence_score(
                    expensive_parse_meta.get("elector_confidence")
                )
                use_expensive_result = (
                    expensive_rank > initial_rank
                    or (expensive_rank == initial_rank and len(expensive_missing_fields) < initial_missing_count)
                    or (
                        expensive_rank == initial_rank
                        and len(expensive_missing_fields) == initial_missing_count
                        and expensive_conf_score > initial_conf_score
                    )
                )
                if use_expensive_result:
                    record = expensive_record
                    parse_status = expensive_status
                    reject_reason = expensive_reason
                    missing_fields = expensive_missing_fields
                    parse_meta = expensive_parse_meta
                    zone_ocr_payload = expensive_zone_ocr
                    raw_card_text = expensive_raw_text
                    cleaned_card_text = expensive_cleaned_text
                    cleaned_body_text = _clean_card_ocr_text(str(expensive_zone_ocr.get("body_text") or ""))
                    ocr_error = expensive_zone_ocr.get("combined_error")

        top_retry_attempted = False
        top_retry_used = False
        retry_expanded_box = expanded_box
        if _should_retry_top_capture(
            parse_status=parse_status,
            missing_fields=missing_fields,
            cleaned_body_text=cleaned_body_text,
        ):
            top_retry_attempted = True
            debug_payload["cards_top_retry_attempted"] = int(debug_payload.get("cards_top_retry_attempted", 0)) + 1
            retry_expanded_box = _expand_box(
                (original_x1, original_y1, original_x2, original_y2),
                width=page_width,
                height=page_height,
                extra_top_ratio=retry_extra_top_ratio,
            )
            if retry_expanded_box != expanded_box:
                retry_card_image = page_image.crop(retry_expanded_box)
                retry_zone_ocr = _ocr_card_zones(
                    retry_card_image,
                    timeout_seconds=ocr_timeout_seconds,
                    use_expensive_passes=True,
                )
                retry_raw_text = str(retry_zone_ocr.get("combined_text") or "")
                retry_cleaned_text = _clean_card_ocr_text(retry_raw_text)
                retry_record, retry_status, retry_reason, retry_missing_fields, retry_parse_meta = _parse_card_record(
                    raw_card_text=retry_raw_text,
                    cleaned_card_text=retry_cleaned_text,
                    serial_zone_text=str(retry_zone_ocr.get("serial_text") or ""),
                    serial_zone_digits_text=str(retry_zone_ocr.get("serial_digits_text") or ""),
                    serial_zone_verify_text=str(retry_zone_ocr.get("serial_verify_text") or ""),
                    elector_zone_text=str(retry_zone_ocr.get("elector_text") or ""),
                    elector_zone_verify_text=str(retry_zone_ocr.get("elector_verify_text") or ""),
                    body_zone_text=str(retry_zone_ocr.get("body_text") or ""),
                    file_name=file_name,
                    file_path=file_path,
                    page_number=page_number,
                    constituency=constituency,
                    section_name=section_name,
                    extraction_method=f"card_ocr_{detection_strategy}_top_retry",
                    slot_index=card_index,
                    expected_slots=len(slot_boxes),
                )
                current_rank = _parse_status_rank(parse_status)
                retry_rank = _parse_status_rank(retry_status)
                if retry_rank > current_rank or (
                    retry_rank == current_rank and len(retry_missing_fields) < len(missing_fields)
                ):
                    top_retry_used = True
                    debug_payload["cards_top_retry_used"] = int(debug_payload.get("cards_top_retry_used", 0)) + 1
                    record = retry_record
                    parse_status = retry_status
                    reject_reason = retry_reason
                    missing_fields = retry_missing_fields
                    parse_meta = retry_parse_meta
                    zone_ocr_payload = retry_zone_ocr
                    raw_card_text = retry_raw_text
                    cleaned_card_text = retry_cleaned_text
                    cleaned_body_text = _clean_card_ocr_text(str(retry_zone_ocr.get("body_text") or ""))
                    ocr_error = retry_zone_ocr.get("combined_error")
                    expanded_box = retry_expanded_box
                    x1, y1, x2, y2 = expanded_box
                    card_image = retry_card_image

        parsed_slot = parse_status in {"valid", "partial_top_missing", "partial_field_missing"}
        insert_candidate = _is_useful_record_for_insert(record, parse_status=parse_status)

        if parse_status == "valid":
            debug_payload["cards_valid"] = int(debug_payload.get("cards_valid", 0)) + 1
        elif parse_status in {"partial_top_missing", "partial_field_missing"}:
            debug_payload["cards_partial"] = int(debug_payload.get("cards_partial", 0)) + 1
            if parse_status == "partial_top_missing":
                debug_payload["cards_missing_top_fields"] = int(debug_payload.get("cards_missing_top_fields", 0)) + 1
        else:
            debug_payload["cards_rejected"] = int(debug_payload.get("cards_rejected", 0)) + 1
            reason_key = (reject_reason or "rejected").strip() or "rejected"
            reject_breakdown = dict(debug_payload.get("reject_reason_breakdown", {}))
            reject_breakdown[reason_key] = int(reject_breakdown.get(reason_key, 0)) + 1
            debug_payload["reject_reason_breakdown"] = reject_breakdown
            failed_slots.append(
                {
                    "slot_index": int(card_index),
                    "parse_status": parse_status,
                    "reason": reason_key,
                    "missing_fields": list(missing_fields),
                }
            )

        if insert_candidate and record is not None:
            parsed_records.append(record)

        if include_card_debug:
            card_debug_entry: dict[str, Any] = {
                "slot_index": card_index,
                "card_index": card_index,
                "original_bbox": {
                    "x1": int(original_x1),
                    "y1": int(original_y1),
                    "x2": int(original_x2),
                    "y2": int(original_y2),
                },
                "expanded_bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "ocr_error": ocr_error,
                "ocr_preprocess": dict(zone_ocr_payload.get("preprocess") or {}),
                "accepted": insert_candidate,
                "reject_reason": reject_reason,
                "parse_status": parse_status,
                "validation_status": parse_status,
                "insert_decision": (
                    "will_insert"
                    if insert_candidate
                    else ("skip_rejected" if not parsed_slot else "skip_partial_not_useful")
                ),
                "db_insert_status": (
                    "pending_insert"
                    if insert_candidate
                    else ("rejected_not_inserted" if not parsed_slot else "partial_not_inserted")
                ),
                "raw_ocr_text": raw_card_text.strip(),
                "cleaned_ocr_text": cleaned_card_text,
                "serial_zone_ocr_text": str(zone_ocr_payload.get("serial_text") or ""),
                "serial_zone_digits_ocr_text": str(zone_ocr_payload.get("serial_digits_text") or ""),
                "serial_zone_verify_ocr_text": str(zone_ocr_payload.get("serial_verify_text") or ""),
                "elector_zone_ocr_text": str(zone_ocr_payload.get("elector_text") or ""),
                "elector_zone_verify_ocr_text": str(zone_ocr_payload.get("elector_verify_text") or ""),
                "body_zone_ocr_text": str(zone_ocr_payload.get("body_text") or ""),
                "cleaned_serial_number": parse_meta.get("serial_number_cleaned"),
                "cleaned_elector_id": parse_meta.get("elector_id_cleaned"),
                "cleaned_elector_id_format": str(parse_meta.get("elector_id_format") or ""),
                "serial_candidates": list(parse_meta.get("serial_candidates") or []),
                "elector_candidates": list(parse_meta.get("elector_candidates") or []),
                "elector_raw_candidates": list(parse_meta.get("elector_raw_candidates") or []),
                "serial_confidence": str(parse_meta.get("serial_confidence") or ""),
                "serial_confidence_reason": str(parse_meta.get("serial_confidence_reason") or ""),
                "elector_confidence": str(parse_meta.get("elector_confidence") or ""),
                "elector_confidence_reason": str(parse_meta.get("elector_confidence_reason") or ""),
                "record_status": str(parse_meta.get("record_status") or ""),
                "record_status_reason": str(parse_meta.get("record_status_reason") or ""),
                "normalized_labels_detected": list(parse_meta.get("normalized_labels_detected") or []),
                "field_parse_quality": dict(parse_meta.get("field_parse_quality") or {}),
                "zone_bboxes": dict(zone_ocr_payload.get("zone_bboxes") or {}),
                "missing_fields": list(missing_fields),
                "top_retry_attempted": bool(top_retry_attempted),
                "top_retry_used": bool(top_retry_used),
                "retry_expanded_bbox": {
                    "x1": int(retry_expanded_box[0]),
                    "y1": int(retry_expanded_box[1]),
                    "x2": int(retry_expanded_box[2]),
                    "y2": int(retry_expanded_box[3]),
                },
                "parsed_record": record,
            }
            crop_buffer = io.BytesIO()
            try:
                card_image.save(crop_buffer, format="PNG")
                card_debug_entry["crop_png_bytes"] = crop_buffer.getvalue()
            except Exception:  # noqa: BLE001
                card_debug_entry["crop_png_bytes"] = None
            cards_debug.append(card_debug_entry)

    debug_payload["cards_parsed"] = int(debug_payload.get("cards_valid", 0)) + int(debug_payload.get("cards_partial", 0))
    debug_payload["cards_valid"] = int(debug_payload.get("cards_valid", 0))
    debug_payload["cards_partial"] = int(debug_payload.get("cards_partial", 0))
    debug_payload["cards_rejected"] = int(debug_payload.get("cards_rejected", 0))
    debug_payload["cards_missing_top_fields"] = int(debug_payload.get("cards_missing_top_fields", 0))
    debug_payload["cards_top_retry_attempted"] = int(debug_payload.get("cards_top_retry_attempted", 0))
    debug_payload["cards_top_retry_used"] = int(debug_payload.get("cards_top_retry_used", 0))
    debug_payload["records_total_for_insert"] = len(parsed_records)
    debug_payload["records_metadata_propagated"] = sum(
        1
        for record in parsed_records
        if str(record.get("constituency") or "").strip() or str(record.get("section_name") or "").strip()
    )
    debug_payload["serial_number_filled_count"] = sum(
        1 for record in parsed_records if str(record.get("serial_number") or "").strip()
    )
    debug_payload["elector_id_valid_count"] = sum(
        1 for record in parsed_records if _is_valid_elector_id(str(record.get("elector_id") or ""))
    )
    debug_payload["elector_id_compact_accepted_count"] = sum(
        1 for record in parsed_records if _elector_id_format(str(record.get("elector_id") or "")) == "compact"
    )
    debug_payload["elector_id_slash_accepted_count"] = sum(
        1 for record in parsed_records if _elector_id_format(str(record.get("elector_id") or "")) == "slash"
    )
    debug_payload["elector_id_trusted_count"] = sum(
        1 for record in parsed_records if str(record.get("_elector_confidence") or "") == "trusted"
    )
    debug_payload["gender_filled_count"] = sum(
        1 for record in parsed_records if str(record.get("gender") or "").strip()
    )
    debug_payload["records_partial_after_cleanup"] = sum(
        1 for record in parsed_records if str(record.get("_record_status") or "") == "partial"
    )
    debug_payload["trusted_records_count"] = sum(
        1 for record in parsed_records if str(record.get("_record_status") or "") == "trusted"
    )
    debug_payload["needs_review_records_count"] = sum(
        1 for record in parsed_records if str(record.get("_record_status") or "") == "needs_review"
    )
    debug_payload["serial_low_confidence_count"] = sum(
        1 for record in parsed_records if str(record.get("_serial_confidence") or "") in {"low_confidence", "needs_review"}
    )
    debug_payload["elector_id_low_confidence_count"] = sum(
        1 for record in parsed_records if str(record.get("_elector_confidence") or "") in {"low_confidence", "needs_review"}
    )
    debug_payload["failed_slots"] = failed_slots
    debug_payload["failed_slot_indexes"] = [int(item["slot_index"]) for item in failed_slots]
    if "reject_reason_breakdown" not in debug_payload:
        debug_payload["reject_reason_breakdown"] = {}
    debug_payload["cards"] = cards_debug
    if not parsed_records and not debug_payload.get("detection_error"):
        debug_payload["detection_error"] = "no valid/partial useful slot records parsed"
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
