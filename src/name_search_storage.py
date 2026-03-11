"""PostgreSQL storage and OCR-structure parsing helpers for PDF Name Search."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote_plus

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve PostgreSQL connection URL from explicit input or environment variables."""

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
