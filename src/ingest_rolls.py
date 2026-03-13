"""Offline batch ingestion for electoral-roll PDFs into PostgreSQL."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import PyPDF2

from .name_finder import discover_pdf_files, run_name_search_progressive
from .name_search_storage import (
    count_document_pages_by_status,
    fetch_document_page_status_map,
    fetch_document_row_by_file_path,
    open_storage_connection,
    parse_voter_records_from_page_layout_aware,
    replace_page_records,
    update_document_status,
    upsert_document,
    upsert_page,
)


TERMINAL_PAGE_STATUSES = {"processed", "skipped"}
ATTEMPTED_PAGE_STATUSES = {"processed", "skipped", "failed"}


@dataclass(frozen=True)
class BatchConfig:
    folder_path: str
    database_url: str | None
    start_page: int
    end_page: int | None
    workers: int
    enable_ocr_fallback: bool
    ocr_timeout_seconds: float | None
    max_files: int | None
    resume: bool
    reprocess_changed: bool
    reprocess_failed: bool


@dataclass(frozen=True)
class FileIngestionResult:
    file_path: str
    file_name: str
    status: str
    reason: str
    pages_total: int
    pages_targeted: int
    pages_attempted: int
    pages_skipped: int
    records_inserted: int
    low_confidence_records: int
    elapsed_seconds: float


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ingest electoral-roll PDFs into PostgreSQL.")
    parser.add_argument("--folder", required=True, help="Folder containing PDFs to ingest recursively.")
    parser.add_argument("--database-url", default="", help="Optional PostgreSQL DATABASE_URL override.")
    parser.add_argument("--start-page", type=int, default=3, help="Start page per file (default: 3).")
    parser.add_argument("--end-page", type=int, default=0, help="End page per file, 0 means no end page.")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent file workers (1-4 recommended).")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap on number of files, 0 means all.")
    parser.add_argument("--ocr-timeout", type=float, default=20.0, help="OCR timeout per page in seconds.")
    parser.add_argument(
        "--disable-ocr-fallback",
        action="store_true",
        help="Disable OCR fallback (only text extractors will run).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and process target page range regardless of existing page status.",
    )
    parser.add_argument(
        "--no-reprocess-changed",
        action="store_true",
        help="If file signature changed, keep existing completed pages and process only missing pages.",
    )
    parser.add_argument(
        "--no-reprocess-failed",
        action="store_true",
        help="When resuming unchanged files, do not retry previously failed pages.",
    )
    return parser.parse_args()


def _safe_file_signature(pdf_path: Path) -> tuple[float | None, int | None]:
    try:
        stat = pdf_path.stat()
    except OSError:
        return None, None
    return float(stat.st_mtime), int(stat.st_size)


def _is_same_signature(existing_row: dict[str, Any] | None, *, mtime: float | None, file_size: int | None) -> bool:
    if not existing_row:
        return False
    existing_mtime = existing_row.get("last_modified")
    existing_size = existing_row.get("file_size")
    if existing_mtime is None or existing_size is None or mtime is None or file_size is None:
        return False
    return abs(float(existing_mtime) - float(mtime)) <= 1e-6 and int(existing_size) == int(file_size)


def _get_pdf_page_count(pdf_path: Path) -> int:
    try:
        with pdf_path.open("rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            return int(len(reader.pages))
    except Exception:
        return 0


def _contiguous_ranges(page_numbers: Sequence[int]) -> list[tuple[int, int]]:
    ordered_pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not ordered_pages:
        return []
    ranges: list[tuple[int, int]] = []
    range_start = ordered_pages[0]
    previous = ordered_pages[0]
    for page_number in ordered_pages[1:]:
        if page_number == previous + 1:
            previous = page_number
            continue
        ranges.append((range_start, previous))
        range_start = page_number
        previous = page_number
    ranges.append((range_start, previous))
    return ranges


def _mark_record_review_status(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    prepared_records: list[dict[str, Any]] = []
    low_confidence_count = 0
    for source_record in records:
        record = dict(source_record)
        record_status = str(record.get("_record_status") or "")
        extraction_method = str(record.get("extraction_method") or "card_ocr_layout")
        if record_status in {"needs_review", "partial"}:
            low_confidence_count += 1
            extraction_method = f"{extraction_method}|{record_status}"
        record["extraction_method"] = extraction_method
        prepared_records.append(record)
    return prepared_records, low_confidence_count


class _StorageLifecycleWriter:
    def __init__(
        self,
        *,
        connection,
        folder_path: str,
        file_path: str,
        file_name: str,
        pages_total: int,
        ocr_timeout_seconds: float | None,
        enable_ocr_fallback: bool,
    ) -> None:
        self.connection = connection
        self.folder_path = folder_path
        self.file_path = file_path
        self.file_name = file_name
        self.pages_total = int(pages_total)
        self.ocr_timeout_seconds = ocr_timeout_seconds
        self.enable_ocr_fallback = bool(enable_ocr_fallback)

        self.document_id = upsert_document(
            self.connection,
            folder_path=self.folder_path,
            file_name=self.file_name,
            file_path=self.file_path,
            pages_total=self.pages_total,
            status="processing",
            error_message=None,
        )
        self.pages_attempted = 0
        self.records_inserted = 0
        self.low_confidence_records = 0
        self.page_errors = 0

    def _refresh_document_pages_processed(self, *, status: str, error_message: str | None = None) -> None:
        attempted_pages_count = count_document_pages_by_status(
            self.connection,
            document_id=self.document_id,
            statuses=sorted(ATTEMPTED_PAGE_STATUSES),
        )
        update_document_status(
            self.connection,
            document_id=self.document_id,
            status=status,
            pages_processed=int(attempted_pages_count),
            error_message=error_message,
        )

    def on_lifecycle(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("event") or "")
        if event_type == "page_started":
            page_number = int(event.get("page_number") or 0)
            if page_number <= 0:
                return
            upsert_page(
                self.connection,
                document_id=self.document_id,
                page_number=page_number,
                status="processing",
                extraction_method=None,
                raw_text="",
                parsed_record_count=0,
                error_message=None,
            )
            return

        if event_type != "page_finished":
            return

        page_number = int(event.get("page_number") or 0)
        if page_number <= 0:
            return
        self.pages_attempted += 1

        page_status = str(event.get("status") or "processed")
        extraction_method = str(event.get("extraction_method") or "")
        raw_text_value = str(event.get("raw_text") or "")
        normalized_text_value = str(event.get("text") or "")
        text_for_storage = raw_text_value if raw_text_value.strip() else normalized_text_value
        error_message = str(event.get("error_message") or "")

        parsed_records: list[dict[str, Any]] = []
        parse_error: str | None = None
        try:
            card_parse_payload = parse_voter_records_from_page_layout_aware(
                page_text=text_for_storage,
                file_name=self.file_name,
                file_path=self.file_path,
                page_number=page_number,
                extraction_method=extraction_method or "exact_text",
                ocr_timeout_seconds=self.ocr_timeout_seconds if self.enable_ocr_fallback else None,
                max_preview_cards=0,
                include_card_debug=False,
            )
            parsed_records = list(card_parse_payload.get("records", []))
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)
            self.page_errors += 1

        if parse_error:
            page_status = "failed"
            error_message = f"layout parser failed: {parse_error}"
            parsed_records = []

        if parsed_records and not text_for_storage.strip():
            text_for_storage = "\n\n".join(
                str(record.get("raw_record_text") or "").strip()
                for record in parsed_records
                if str(record.get("raw_record_text") or "").strip()
            )
        if parsed_records and page_status != "processed":
            page_status = "processed"
        if parsed_records and not extraction_method:
            extraction_method = "card_ocr_layout"

        prepared_records, page_low_confidence = _mark_record_review_status(parsed_records)

        try:
            page_id = upsert_page(
                self.connection,
                document_id=self.document_id,
                page_number=page_number,
                status=page_status,
                extraction_method=extraction_method or None,
                raw_text=text_for_storage,
                parsed_record_count=len(prepared_records),
                error_message=error_message or None,
            )
            replace_page_records(
                self.connection,
                document_id=self.document_id,
                page_id=page_id,
                records=prepared_records,
                page_number=page_number,
                extraction_method=None,
            )
            self.records_inserted += len(prepared_records)
            self.low_confidence_records += int(page_low_confidence)
            self._refresh_document_pages_processed(status="processing")
        except Exception as exc:  # noqa: BLE001
            self.page_errors += 1
            upsert_page(
                self.connection,
                document_id=self.document_id,
                page_number=page_number,
                status="failed",
                extraction_method=extraction_method or None,
                raw_text=text_for_storage,
                parsed_record_count=0,
                error_message=str(exc),
            )
            self._refresh_document_pages_processed(status="processing", error_message=str(exc))


def _process_single_file(config: BatchConfig, *, file_path: Path) -> FileIngestionResult:
    start_time = time.perf_counter()
    connection = None
    file_name = file_path.name
    resolved_file_path = str(file_path.resolve())
    try:
        connection = open_storage_connection(config.database_url)
        mtime, file_size = _safe_file_signature(file_path)
        existing_row = fetch_document_row_by_file_path(connection, file_path=resolved_file_path)
        unchanged_signature = _is_same_signature(existing_row, mtime=mtime, file_size=file_size)

        page_count = _get_pdf_page_count(file_path)
        if page_count <= 0:
            return FileIngestionResult(
                file_path=resolved_file_path,
                file_name=file_name,
                status="failed",
                reason="failed to determine PDF page count",
                pages_total=0,
                pages_targeted=0,
                pages_attempted=0,
                pages_skipped=0,
                records_inserted=0,
                low_confidence_records=0,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        start_page = max(1, int(config.start_page))
        end_page = int(config.end_page) if config.end_page is not None else int(page_count)
        end_page = min(end_page, page_count)
        if end_page < start_page:
            return FileIngestionResult(
                file_path=resolved_file_path,
                file_name=file_name,
                status="skipped",
                reason=f"no target pages in range start={start_page}, end={end_page}",
                pages_total=page_count,
                pages_targeted=0,
                pages_attempted=0,
                pages_skipped=0,
                records_inserted=0,
                low_confidence_records=0,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        target_pages = list(range(start_page, end_page + 1))
        target_count = len(target_pages)

        document_id = upsert_document(
            connection,
            folder_path=config.folder_path,
            file_name=file_name,
            file_path=resolved_file_path,
            pages_total=page_count,
            status="pending",
            error_message=None,
        )

        if existing_row and not unchanged_signature and config.reprocess_changed:
            connection.execute(
                "DELETE FROM pages WHERE document_id = %s AND page_number BETWEEN %s AND %s",
                (int(document_id), int(start_page), int(end_page)),
            )
            connection.execute(
                "DELETE FROM pages WHERE document_id = %s AND page_number > %s",
                (int(document_id), int(page_count)),
            )
            connection.commit()

        pending_pages = list(target_pages)
        if config.resume:
            status_map = fetch_document_page_status_map(
                connection,
                document_id=int(document_id),
                start_page=start_page,
                end_page=end_page,
            )
            pending_pages = []
            for page_number in target_pages:
                page_status = str(status_map.get(page_number) or "")
                if page_status in TERMINAL_PAGE_STATUSES:
                    continue
                if page_status == "failed" and not config.reprocess_failed:
                    continue
                pending_pages.append(page_number)

        completed_before = target_count - len(pending_pages)
        if (
            unchanged_signature
            and existing_row
            and str(existing_row.get("status") or "") == "processed"
            and not pending_pages
        ):
            attempted_count = count_document_pages_by_status(
                connection,
                document_id=int(document_id),
                statuses=sorted(ATTEMPTED_PAGE_STATUSES),
            )
            update_document_status(
                connection,
                document_id=int(document_id),
                status="processed",
                pages_processed=int(attempted_count),
                error_message=None,
            )
            return FileIngestionResult(
                file_path=resolved_file_path,
                file_name=file_name,
                status="skipped",
                reason="unchanged file already completed",
                pages_total=page_count,
                pages_targeted=target_count,
                pages_attempted=0,
                pages_skipped=target_count,
                records_inserted=0,
                low_confidence_records=0,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        if not pending_pages:
            attempted_count = count_document_pages_by_status(
                connection,
                document_id=int(document_id),
                statuses=sorted(ATTEMPTED_PAGE_STATUSES),
            )
            update_document_status(
                connection,
                document_id=int(document_id),
                status="processed",
                pages_processed=int(attempted_count),
                error_message=None,
            )
            return FileIngestionResult(
                file_path=resolved_file_path,
                file_name=file_name,
                status="processed",
                reason="no pending pages after checkpoint scan",
                pages_total=page_count,
                pages_targeted=target_count,
                pages_attempted=0,
                pages_skipped=target_count,
                records_inserted=0,
                low_confidence_records=0,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        update_document_status(
            connection,
            document_id=int(document_id),
            status="processing",
            pages_processed=count_document_pages_by_status(
                connection,
                document_id=int(document_id),
                statuses=sorted(ATTEMPTED_PAGE_STATUSES),
            ),
            error_message=None,
        )

        lifecycle_writer = _StorageLifecycleWriter(
            connection=connection,
            folder_path=config.folder_path,
            file_path=resolved_file_path,
            file_name=file_name,
            pages_total=page_count,
            ocr_timeout_seconds=config.ocr_timeout_seconds,
            enable_ocr_fallback=config.enable_ocr_fallback,
        )
        ranges = _contiguous_ranges(pending_pages)
        for range_start, range_end in ranges:
            run_name_search_progressive(
                folder_path=config.folder_path,
                raw_names="__process_and_store_only__",
                start_page=int(range_start),
                end_page=int(range_end),
                enable_ocr_fallback=bool(config.enable_ocr_fallback),
                ocr_timeout_per_page=config.ocr_timeout_seconds if config.enable_ocr_fallback else None,
                overall_timeout_seconds=None,
                pdf_files_override=[resolved_file_path],
                max_files=1,
                max_pages_per_file=None,
                progress_callback=None,
                lifecycle_callback=lifecycle_writer.on_lifecycle,
                stop_check=None,
            )

        terminal_count = count_document_pages_by_status(
            connection,
            document_id=int(document_id),
            statuses=sorted(TERMINAL_PAGE_STATUSES),
            start_page=start_page,
            end_page=end_page,
        )
        failed_count = count_document_pages_by_status(
            connection,
            document_id=int(document_id),
            statuses=["failed"],
            start_page=start_page,
            end_page=end_page,
        )
        attempted_count = count_document_pages_by_status(
            connection,
            document_id=int(document_id),
            statuses=sorted(ATTEMPTED_PAGE_STATUSES),
        )

        final_status = "processed"
        final_reason = "completed target page range"
        if failed_count > 0:
            final_status = "failed"
            final_reason = f"{failed_count} page(s) failed in target range"
        elif terminal_count < target_count:
            final_status = "processing"
            final_reason = "pending pages remain in target range"

        update_document_status(
            connection,
            document_id=int(document_id),
            status=final_status,
            pages_processed=int(attempted_count),
            error_message=None if final_status == "processed" else final_reason,
        )

        return FileIngestionResult(
            file_path=resolved_file_path,
            file_name=file_name,
            status=final_status,
            reason=final_reason,
            pages_total=page_count,
            pages_targeted=target_count,
            pages_attempted=int(lifecycle_writer.pages_attempted),
            pages_skipped=int(completed_before),
            records_inserted=int(lifecycle_writer.records_inserted),
            low_confidence_records=int(lifecycle_writer.low_confidence_records),
            elapsed_seconds=time.perf_counter() - start_time,
        )
    except Exception as exc:  # noqa: BLE001
        if connection is not None:
            try:
                existing = fetch_document_row_by_file_path(connection, file_path=resolved_file_path)
                if existing:
                    attempted_count = count_document_pages_by_status(
                        connection,
                        document_id=int(existing["id"]),
                        statuses=sorted(ATTEMPTED_PAGE_STATUSES),
                    )
                    update_document_status(
                        connection,
                        document_id=int(existing["id"]),
                        status="failed",
                        pages_processed=int(attempted_count),
                        error_message=str(exc),
                    )
            except Exception:
                pass
        return FileIngestionResult(
            file_path=resolved_file_path,
            file_name=file_name,
            status="failed",
            reason=str(exc),
            pages_total=0,
            pages_targeted=0,
            pages_attempted=0,
            pages_skipped=0,
            records_inserted=0,
            low_confidence_records=0,
            elapsed_seconds=time.perf_counter() - start_time,
        )
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def main() -> int:
    args = _parse_args()

    folder_path = Path(args.folder).expanduser().resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        _log(f"Invalid folder: {folder_path}")
        return 2

    discovered_files = discover_pdf_files(folder_path)
    max_files = int(args.max_files) if int(args.max_files) > 0 else None
    if max_files is not None:
        discovered_files = discovered_files[:max_files]

    if not discovered_files:
        _log("No PDF files discovered.")
        return 0

    requested_workers = max(1, int(args.workers))
    worker_count = min(requested_workers, 4)
    if requested_workers != worker_count:
        _log(f"Workers capped to {worker_count} (requested {requested_workers}).")

    config = BatchConfig(
        folder_path=str(folder_path),
        database_url=args.database_url.strip() or None,
        start_page=max(1, int(args.start_page)),
        end_page=(int(args.end_page) if int(args.end_page) > 0 else None),
        workers=worker_count,
        enable_ocr_fallback=not bool(args.disable_ocr_fallback),
        ocr_timeout_seconds=(float(args.ocr_timeout) if float(args.ocr_timeout) > 0 else None),
        max_files=max_files,
        resume=not bool(args.no_resume),
        reprocess_changed=not bool(args.no_reprocess_changed),
        reprocess_failed=not bool(args.no_reprocess_failed),
    )

    _log(
        "Batch ingestion starting: "
        + f"files={len(discovered_files)}, workers={config.workers}, "
        + f"resume={'yes' if config.resume else 'no'}, "
        + f"ocr_fallback={'yes' if config.enable_ocr_fallback else 'no'}"
    )
    start_time = time.perf_counter()

    aggregate = {
        "files_discovered": len(discovered_files),
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "pages_attempted": 0,
        "pages_skipped": 0,
        "records_inserted": 0,
        "low_confidence_records": 0,
    }

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(_process_single_file, config, file_path=pdf_path): pdf_path
            for pdf_path in discovered_files
        }
        for future in as_completed(futures):
            result = future.result()
            aggregate["pages_attempted"] += int(result.pages_attempted)
            aggregate["pages_skipped"] += int(result.pages_skipped)
            aggregate["records_inserted"] += int(result.records_inserted)
            aggregate["low_confidence_records"] += int(result.low_confidence_records)

            if result.status == "failed":
                aggregate["files_failed"] += 1
            elif result.status == "skipped":
                aggregate["files_skipped"] += 1
            else:
                aggregate["files_processed"] += 1

            _log(
                f"{result.file_name}: status={result.status}, "
                + f"target_pages={result.pages_targeted}, attempted={result.pages_attempted}, "
                + f"skipped={result.pages_skipped}, inserted={result.records_inserted}, "
                + f"low_conf={result.low_confidence_records}, reason={result.reason}"
            )

    elapsed = time.perf_counter() - start_time
    _log(
        "Batch ingestion complete: "
        + f"files_discovered={aggregate['files_discovered']}, "
        + f"files_processed={aggregate['files_processed']}, "
        + f"files_skipped={aggregate['files_skipped']}, "
        + f"files_failed={aggregate['files_failed']}, "
        + f"pages_attempted={aggregate['pages_attempted']}, "
        + f"pages_skipped={aggregate['pages_skipped']}, "
        + f"records_inserted={aggregate['records_inserted']}, "
        + f"low_confidence_records={aggregate['low_confidence_records']}, "
        + f"elapsed={elapsed:.1f}s"
    )
    return 0 if int(aggregate["files_failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
