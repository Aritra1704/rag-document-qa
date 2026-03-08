"""Folder-based PDF name verification utilities."""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import PyPDF2


CSV_COLUMNS = [
    "searched_name",
    "file_name",
    "file_path",
    "page_number",
    "snippet",
    "match_type",
]

EXTRACTOR_ORDER = ["PyPDF2", "pypdf", "pdfplumber", "pymupdf", "pdftotext"]


@dataclass(frozen=True)
class PageRecord:
    file_name: str
    file_path: str
    page_number: int
    text: str


@dataclass(frozen=True)
class NameMatch:
    searched_name: str
    file_name: str
    file_path: str
    page_number: int
    match_position: int
    snippet: str
    match_type: str = "exact"


@dataclass
class NameSearchOutcome:
    folder_path: str
    names: List[str]
    pdf_files: List[str]
    skipped_files: List[str]
    results: List[NameMatch]
    extraction_debug: List["FileExtractionDebug"] = field(default_factory=list)

    @property
    def names_without_matches(self) -> List[str]:
        matched_names = {match.searched_name for match in self.results}
        return [name for name in self.names if name not in matched_names]


@dataclass
class _PdfExtractor:
    name: str
    page_count: int
    extract_page_text: Callable[[int], Tuple[str, str | None]]
    close: Callable[[], None]


@dataclass(frozen=True)
class ExtractorOpenDebug:
    extractor_name: str
    import_available: bool
    open_attempted: bool
    open_succeeded: bool
    error: str | None = None


@dataclass(frozen=True)
class ExtractorAttemptDebug:
    extractor_name: str
    import_available: bool
    open_attempted: bool
    extraction_attempted: bool
    succeeded: bool
    character_count: int
    whitespace_only: bool
    preview: str
    error: str | None = None


@dataclass(frozen=True)
class PageExtractionDebug:
    file_path: str
    page_number: int
    attempted_extractors: List[str]
    successful_extractor: str | None
    character_count: int
    whitespace_only: bool
    skipped: bool
    preview: str
    attempts: List[ExtractorAttemptDebug]


@dataclass(frozen=True)
class FileExtractionDebug:
    file_name: str
    file_path: str
    page_debug: List[PageExtractionDebug]
    extractor_open_debug: List[ExtractorOpenDebug]
    skipped: bool
    skip_reason: str | None


def parse_names(raw_names: str | Sequence[str]) -> List[str]:
    """Parse comma-separated or sequence input into unique, cleaned names."""

    if isinstance(raw_names, str):
        candidates = raw_names.split(",")
    else:
        candidates = list(raw_names)

    cleaned_names: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = " ".join(str(candidate).strip().split())
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned_names.append(normalized)
    return cleaned_names


def discover_pdf_files(folder_path: str | Path) -> List[Path]:
    """Recursively discover PDFs under a local folder."""

    resolved_folder = Path(folder_path).expanduser()
    if not resolved_folder.exists() or not resolved_folder.is_dir():
        raise ValueError(f"Invalid folder path: {resolved_folder}")

    pdf_files: List[Path] = []
    for root, _, files in os.walk(resolved_folder):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                pdf_files.append((Path(root) / file_name).resolve())

    return sorted(pdf_files, key=lambda item: str(item).lower())


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _preview_text(text: str, max_chars: int = 160) -> str:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + " ..."


def _open_pypdf2_extractor(pdf_path: Path) -> Tuple[_PdfExtractor | None, ExtractorOpenDebug]:
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="PyPDF2",
            import_available=True,
            open_attempted=True,
            open_succeeded=False,
            error=f"open failed: {exc}",
        )

    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except Exception:  # noqa: BLE001
            unlocked = 0
        if unlocked == 0:
            return None, ExtractorOpenDebug(
                extractor_name="PyPDF2",
                import_available=True,
                open_attempted=True,
                open_succeeded=False,
                error="password-protected",
            )

    def extract_page_text(page_index: int) -> Tuple[str, str | None]:
        try:
            return reader.pages[page_index].extract_text() or "", None
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)

    return (
        _PdfExtractor(
            name="PyPDF2",
            page_count=len(reader.pages),
            extract_page_text=extract_page_text,
            close=lambda: None,
        ),
        ExtractorOpenDebug(
            extractor_name="PyPDF2",
            import_available=True,
            open_attempted=True,
            open_succeeded=True,
            error=None,
        ),
    )


def _open_pypdf_extractor(pdf_path: Path) -> Tuple[_PdfExtractor | None, ExtractorOpenDebug]:
    try:
        import pypdf
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="pypdf",
            import_available=False,
            open_attempted=False,
            open_succeeded=False,
            error=f"import failed: {exc}",
        )

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="pypdf",
            import_available=True,
            open_attempted=True,
            open_succeeded=False,
            error=f"open failed: {exc}",
        )

    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except Exception:  # noqa: BLE001
            unlocked = 0
        if unlocked == 0:
            return None, ExtractorOpenDebug(
                extractor_name="pypdf",
                import_available=True,
                open_attempted=True,
                open_succeeded=False,
                error="password-protected",
            )

    def extract_page_text(page_index: int) -> Tuple[str, str | None]:
        try:
            return reader.pages[page_index].extract_text() or "", None
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)

    return (
        _PdfExtractor(
            name="pypdf",
            page_count=len(reader.pages),
            extract_page_text=extract_page_text,
            close=lambda: None,
        ),
        ExtractorOpenDebug(
            extractor_name="pypdf",
            import_available=True,
            open_attempted=True,
            open_succeeded=True,
            error=None,
        ),
    )


def _open_pdfplumber_extractor(pdf_path: Path) -> Tuple[_PdfExtractor | None, ExtractorOpenDebug]:
    try:
        import pdfplumber
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="pdfplumber",
            import_available=False,
            open_attempted=False,
            open_succeeded=False,
            error=f"import failed: {exc}",
        )

    try:
        pdf_doc = pdfplumber.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="pdfplumber",
            import_available=True,
            open_attempted=True,
            open_succeeded=False,
            error=f"open failed: {exc}",
        )

    def extract_page_text(page_index: int) -> Tuple[str, str | None]:
        try:
            return pdf_doc.pages[page_index].extract_text() or "", None
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)

    def close() -> None:
        try:
            pdf_doc.close()
        except Exception:  # noqa: BLE001
            pass

    return (
        _PdfExtractor(
            name="pdfplumber",
            page_count=len(pdf_doc.pages),
            extract_page_text=extract_page_text,
            close=close,
        ),
        ExtractorOpenDebug(
            extractor_name="pdfplumber",
            import_available=True,
            open_attempted=True,
            open_succeeded=True,
            error=None,
        ),
    )


def _open_pymupdf_extractor(pdf_path: Path) -> Tuple[_PdfExtractor | None, ExtractorOpenDebug]:
    try:
        import fitz
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="pymupdf",
            import_available=False,
            open_attempted=False,
            open_succeeded=False,
            error=f"import failed: {exc}",
        )

    try:
        pdf_doc = fitz.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="pymupdf",
            import_available=True,
            open_attempted=True,
            open_succeeded=False,
            error=f"open failed: {exc}",
        )

    if getattr(pdf_doc, "needs_pass", False):
        try:
            unlocked = pdf_doc.authenticate("")
        except Exception:  # noqa: BLE001
            unlocked = False
        if not unlocked:
            try:
                pdf_doc.close()
            except Exception:  # noqa: BLE001
                pass
            return None, ExtractorOpenDebug(
                extractor_name="pymupdf",
                import_available=True,
                open_attempted=True,
                open_succeeded=False,
                error="password-protected",
            )

    def extract_page_text(page_index: int) -> Tuple[str, str | None]:
        try:
            page = pdf_doc.load_page(page_index)
            return page.get_text("text") or "", None
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)

    def close() -> None:
        try:
            pdf_doc.close()
        except Exception:  # noqa: BLE001
            pass

    return (
        _PdfExtractor(
            name="pymupdf",
            page_count=int(pdf_doc.page_count),
            extract_page_text=extract_page_text,
            close=close,
        ),
        ExtractorOpenDebug(
            extractor_name="pymupdf",
            import_available=True,
            open_attempted=True,
            open_succeeded=True,
            error=None,
        ),
    )


def _open_pdftotext_extractor(pdf_path: Path) -> Tuple[_PdfExtractor | None, ExtractorOpenDebug]:
    pdftotext_path = shutil.which("pdftotext")
    if not pdftotext_path:
        return None, ExtractorOpenDebug(
            extractor_name="pdftotext",
            import_available=False,
            open_attempted=False,
            open_succeeded=False,
            error="command not found",
        )

    command = [
        pdftotext_path,
        "-enc",
        "UTF-8",
        "-layout",
        "-q",
        str(pdf_path),
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        return None, ExtractorOpenDebug(
            extractor_name="pdftotext",
            import_available=True,
            open_attempted=True,
            open_succeeded=False,
            error=f"command failed: {exc}",
        )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return None, ExtractorOpenDebug(
            extractor_name="pdftotext",
            import_available=True,
            open_attempted=True,
            open_succeeded=False,
            error=f"pdftotext exit {result.returncode}: {stderr or 'no stderr'}",
        )

    page_texts = result.stdout.split("\f")
    if page_texts and page_texts[-1] == "":
        page_texts = page_texts[:-1]
    if not page_texts:
        page_texts = [""]

    def extract_page_text(page_index: int) -> Tuple[str, str | None]:
        if page_index < 0 or page_index >= len(page_texts):
            return "", "page unavailable for pdftotext output"
        return page_texts[page_index], None

    return (
        _PdfExtractor(
            name="pdftotext",
            page_count=len(page_texts),
            extract_page_text=extract_page_text,
            close=lambda: None,
        ),
        ExtractorOpenDebug(
            extractor_name="pdftotext",
            import_available=True,
            open_attempted=True,
            open_succeeded=True,
            error=None,
        ),
    )


def _build_pdf_extractors(
    pdf_path: Path,
) -> Tuple[dict[str, _PdfExtractor], dict[str, ExtractorOpenDebug]]:
    extractors: dict[str, _PdfExtractor] = {}
    open_debug: dict[str, ExtractorOpenDebug] = {}

    for extractor_name, opener in (
        ("PyPDF2", _open_pypdf2_extractor),
        ("pypdf", _open_pypdf_extractor),
        ("pdfplumber", _open_pdfplumber_extractor),
        ("pymupdf", _open_pymupdf_extractor),
        ("pdftotext", _open_pdftotext_extractor),
    ):
        extractor, extractor_debug = opener(pdf_path)
        open_debug[extractor_name] = extractor_debug
        if extractor:
            extractors[extractor_name] = extractor

    return extractors, open_debug


def collect_pdf_pages(
    pdf_files: Sequence[Path],
    include_debug: bool = False,
    max_pages_per_file: int | None = None,
) -> Tuple[List[PageRecord], List[str]] | Tuple[List[PageRecord], List[str], List[FileExtractionDebug]]:
    """Extract page-level text from PDF files, skipping unreadable inputs."""

    page_records: List[PageRecord] = []
    skipped_files: List[str] = []
    file_debug_entries: List[FileExtractionDebug] = []
    extractor_order = EXTRACTOR_ORDER

    for pdf_path in pdf_files:
        extractors, open_debug = _build_pdf_extractors(pdf_path)
        if not extractors:
            error_context = "; ".join(
                debug.error for debug in open_debug.values() if debug.error
            ) or "unknown error"
            skip_reason = f"unreadable by all extractors: {error_context}"
            skipped_files.append(f"{pdf_path} ({skip_reason})")
            if include_debug:
                file_debug_entries.append(
                    FileExtractionDebug(
                        file_name=pdf_path.name,
                        file_path=str(pdf_path),
                        page_debug=[],
                        extractor_open_debug=[open_debug[name] for name in extractor_order if name in open_debug],
                        skipped=True,
                        skip_reason=skip_reason,
                    )
                )
            continue

        file_has_text = False
        max_page_count = max(extractor.page_count for extractor in extractors.values())
        if max_pages_per_file is not None:
            max_page_count = min(max_page_count, max_pages_per_file)
        page_debug_entries: List[PageExtractionDebug] = []

        try:
            for page_index in range(max_page_count):
                normalized_text = ""
                successful_extractor: str | None = None
                attempt_debug: List[ExtractorAttemptDebug] = []

                for extractor_name in extractor_order:
                    extractor_status = open_debug.get(extractor_name)
                    extractor = extractors.get(extractor_name)
                    if extractor_status is None:
                        extractor_status = ExtractorOpenDebug(
                            extractor_name=extractor_name,
                            import_available=False,
                            open_attempted=False,
                            open_succeeded=False,
                            error="not configured",
                        )

                    if successful_extractor:
                        attempt_debug.append(
                            ExtractorAttemptDebug(
                                extractor_name=extractor_name,
                                import_available=extractor_status.import_available,
                                open_attempted=extractor_status.open_attempted,
                                extraction_attempted=False,
                                succeeded=False,
                                character_count=0,
                                whitespace_only=True,
                                preview="",
                                error=f"skipped after winner: {successful_extractor}",
                            )
                        )
                        continue

                    if not extractor:
                        attempt_debug.append(
                            ExtractorAttemptDebug(
                                extractor_name=extractor_name,
                                import_available=extractor_status.import_available,
                                open_attempted=extractor_status.open_attempted,
                                extraction_attempted=False,
                                succeeded=False,
                                character_count=0,
                                whitespace_only=True,
                                preview="",
                                error=extractor_status.error or "not available",
                            )
                        )
                        continue

                    if page_index >= extractor.page_count:
                        attempt_debug.append(
                            ExtractorAttemptDebug(
                                extractor_name=extractor_name,
                                import_available=extractor_status.import_available,
                                open_attempted=extractor_status.open_attempted,
                                extraction_attempted=False,
                                succeeded=False,
                                character_count=0,
                                whitespace_only=True,
                                preview="",
                                error="page unavailable for this extractor",
                            )
                        )
                        continue

                    raw_text, extract_error = extractor.extract_page_text(page_index)
                    normalized_text = _normalize_whitespace(raw_text)
                    character_count = len(normalized_text)
                    is_whitespace_only = character_count == 0

                    if extract_error:
                        attempt_debug.append(
                            ExtractorAttemptDebug(
                                extractor_name=extractor_name,
                                import_available=extractor_status.import_available,
                                open_attempted=extractor_status.open_attempted,
                                extraction_attempted=True,
                                succeeded=False,
                                character_count=0,
                                whitespace_only=True,
                                preview="",
                                error=extract_error,
                            )
                        )
                        continue

                    preview = _preview_text(normalized_text)
                    attempt_debug.append(
                        ExtractorAttemptDebug(
                            extractor_name=extractor_name,
                            import_available=extractor_status.import_available,
                            open_attempted=extractor_status.open_attempted,
                            extraction_attempted=True,
                            succeeded=not is_whitespace_only,
                            character_count=character_count,
                            whitespace_only=is_whitespace_only,
                            preview=preview,
                            error="empty/whitespace text" if is_whitespace_only else None,
                        )
                    )
                    if normalized_text:
                        successful_extractor = extractor_name

                extracted = bool(normalized_text)
                page_debug_entries.append(
                    PageExtractionDebug(
                        file_path=str(pdf_path),
                        page_number=page_index + 1,
                        attempted_extractors=[attempt.extractor_name for attempt in attempt_debug],
                        successful_extractor=successful_extractor,
                        character_count=len(normalized_text),
                        whitespace_only=not extracted,
                        skipped=not extracted,
                        preview=_preview_text(normalized_text),
                        attempts=attempt_debug,
                    )
                )

                if not normalized_text:
                    continue

                file_has_text = True
                page_records.append(
                    PageRecord(
                        file_name=pdf_path.name,
                        file_path=str(pdf_path),
                        page_number=page_index + 1,
                        text=normalized_text,
                    )
                )
        finally:
            for extractor in extractors.values():
                extractor.close()

        file_skip_reason: str | None = None
        if not file_has_text:
            file_skip_reason = "no extractable text after PyPDF2, pypdf, pdfplumber, pymupdf, and pdftotext"
            skipped_files.append(f"{pdf_path} ({file_skip_reason})")

        if include_debug:
            file_debug_entries.append(
                FileExtractionDebug(
                    file_name=pdf_path.name,
                    file_path=str(pdf_path),
                    page_debug=page_debug_entries,
                    extractor_open_debug=[open_debug[name] for name in extractor_order if name in open_debug],
                    skipped=not file_has_text,
                    skip_reason=file_skip_reason,
                )
            )

    if include_debug:
        return page_records, skipped_files, file_debug_entries

    return page_records, skipped_files


def summarize_extraction_debug(file_debug_entries: Sequence[FileExtractionDebug]) -> dict[str, Any]:
    extractor_success_counts: dict[str, int] = defaultdict(int)
    total_pages_processed = 0
    pages_with_text = 0
    pages_with_no_text = 0
    pdfs_with_extracted_text = 0

    for file_debug in file_debug_entries:
        file_has_extracted_page = False
        for page in file_debug.page_debug:
            total_pages_processed += 1
            if page.skipped:
                pages_with_no_text += 1
                continue
            pages_with_text += 1
            file_has_extracted_page = True
            if page.successful_extractor:
                extractor_success_counts[page.successful_extractor] += 1

        if file_has_extracted_page:
            pdfs_with_extracted_text += 1

    total_pdfs = len(file_debug_entries)
    return {
        "pdfs_discovered": total_pdfs,
        "pdfs_with_extracted_text": pdfs_with_extracted_text,
        "pdfs_fully_skipped": total_pdfs - pdfs_with_extracted_text,
        "total_pages_processed": total_pages_processed,
        "pages_with_text": pages_with_text,
        "pages_with_no_text": pages_with_no_text,
        "extractor_success_counts": dict(extractor_success_counts),
    }


def _build_name_pattern(name: str) -> re.Pattern[str]:
    escaped_name = re.escape(name).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped_name}(?!\w)", flags=re.IGNORECASE)


def _build_snippet(text: str, start: int, end: int, window: int = 80) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].strip()
    if left > 0:
        snippet = f"... {snippet}"
    if right < len(text):
        snippet = f"{snippet} ..."
    return snippet


def _dedupe_matches(matches: Iterable[NameMatch]) -> List[NameMatch]:
    deduped: List[NameMatch] = []
    seen = set()
    for match in matches:
        key = (
            match.searched_name.lower(),
            match.file_path,
            match.page_number,
            match.match_position,
            match.snippet.lower(),
            match.match_type,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def find_exact_name_matches(page_records: Sequence[PageRecord], names: Sequence[str]) -> List[NameMatch]:
    """Find case-insensitive exact name matches page-by-page."""

    patterns = {name: _build_name_pattern(name) for name in names}
    exact_matches: List[NameMatch] = []

    for page_record in page_records:
        for name, pattern in patterns.items():
            for match in pattern.finditer(page_record.text):
                exact_matches.append(
                    NameMatch(
                        searched_name=name,
                        file_name=page_record.file_name,
                        file_path=page_record.file_path,
                        page_number=page_record.page_number,
                        match_position=match.start(),
                        snippet=_build_snippet(page_record.text, match.start(), match.end()),
                        match_type="exact",
                    )
                )

    return _dedupe_matches(exact_matches)


def _build_semantic_snippet(text: str, name: str) -> str:
    pattern = _build_name_pattern(name)
    semantic_match = pattern.search(text)
    if semantic_match:
        return _build_snippet(text, semantic_match.start(), semantic_match.end())
    return _build_snippet(text, 0, min(1, len(text)))


def find_semantic_matches(
    page_records: Sequence[PageRecord],
    names: Sequence[str],
    per_name_limit: int = 5,
) -> Tuple[List[NameMatch], str | None]:
    """Run semantic search over page records and return match-labeled results."""

    if not page_records or not names:
        return [], None

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except Exception as exc:  # noqa: BLE001
        return [], f"Semantic search unavailable: {exc}"

    collection_name = f"name_finder_{uuid.uuid4().hex}"
    semantic_matches: List[NameMatch] = []
    client = chromadb.Client()

    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )

        documents = [record.text for record in page_records]
        metadatas = [
            {
                "file_name": record.file_name,
                "file_path": record.file_path,
                "page_number": record.page_number,
            }
            for record in page_records
        ]
        ids = [f"page_{index}" for index in range(len(page_records))]
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

        result_limit = min(per_name_limit, len(page_records))
        for name in names:
            query_result = collection.query(query_texts=[name], n_results=result_limit)
            documents_for_name = query_result.get("documents", [[]])[0]
            metadatas_for_name = query_result.get("metadatas", [[]])[0]

            for document_text, metadata in zip(documents_for_name, metadatas_for_name):
                if not document_text or not metadata:
                    continue
                semantic_matches.append(
                    NameMatch(
                        searched_name=name,
                        file_name=str(metadata.get("file_name", "")),
                        file_path=str(metadata.get("file_path", "")),
                        page_number=int(metadata.get("page_number", 0)),
                        match_position=-1,
                        snippet=_build_semantic_snippet(str(document_text), name),
                        match_type="semantic",
                    )
                )

        return _dedupe_matches(semantic_matches), None
    except Exception as exc:  # noqa: BLE001
        return [], f"Semantic search unavailable: {exc}"
    finally:
        try:
            client.delete_collection(name=collection_name)
        except Exception:  # noqa: BLE001
            pass


def run_name_search(
    folder_path: str | Path,
    raw_names: str | Sequence[str],
    enable_semantic_fallback: bool = False,
    semantic_mode: str = "fallback",
) -> NameSearchOutcome:
    """
    Run exact search and optional semantic fallback for names across local PDFs.

    semantic_mode:
    - fallback: semantic search only for names without exact matches
    - always: semantic search for all names when enabled
    """

    names = parse_names(raw_names)
    if not names:
        raise ValueError("Please provide at least one name to search.")

    pdf_files = discover_pdf_files(folder_path)
    page_records, skipped_files, extraction_debug = collect_pdf_pages(
        pdf_files,
        include_debug=True,
    )

    exact_matches = find_exact_name_matches(page_records, names)
    combined_matches: List[NameMatch] = list(exact_matches)

    if enable_semantic_fallback and page_records:
        exact_match_names = {match.searched_name for match in exact_matches}
        if semantic_mode == "always":
            semantic_names = names
        else:
            semantic_names = [name for name in names if name not in exact_match_names]

        semantic_matches, semantic_error = find_semantic_matches(page_records, semantic_names)
        if semantic_error:
            skipped_files.append(semantic_error)
        combined_matches.extend(semantic_matches)

    return NameSearchOutcome(
        folder_path=str(Path(folder_path).expanduser().resolve()),
        names=names,
        pdf_files=[str(path) for path in pdf_files],
        skipped_files=skipped_files,
        results=_dedupe_matches(combined_matches),
        extraction_debug=extraction_debug,
    )


def export_results_to_csv(results: Sequence[NameMatch], output_path: str | Path = "results.csv") -> Path:
    """Export name search results to CSV and return the saved path."""

    resolved_path = Path(output_path).expanduser().resolve()
    with resolved_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for match in results:
            writer.writerow(
                {
                    "searched_name": match.searched_name,
                    "file_name": match.file_name,
                    "file_path": match.file_path,
                    "page_number": match.page_number,
                    "snippet": match.snippet,
                    "match_type": match.match_type,
                }
            )
    return resolved_path


def results_grouped_by_name(results: Sequence[NameMatch]) -> dict[str, List[NameMatch]]:
    grouped_results: dict[str, List[NameMatch]] = defaultdict(list)
    for match in results:
        grouped_results[match.searched_name].append(match)
    return grouped_results


def _print_cli_results(outcome: NameSearchOutcome) -> None:
    print(f"\nPDFs discovered: {len(outcome.pdf_files)}")
    print(f"Total matches: {len(outcome.results)}")

    if outcome.skipped_files:
        print("\nSkipped/Warnings:")
        for skipped in outcome.skipped_files:
            print(f"- {skipped}")

    grouped_results = results_grouped_by_name(outcome.results)
    for name in outcome.names:
        print("\n" + "=" * 70)
        print(f"Results for: {name}")
        print("=" * 70)

        matches = grouped_results.get(name, [])
        if not matches:
            print(f"No matches found for: {name}")
            continue

        for match in matches:
            print(f"Name: {match.searched_name}")
            print(f"File: {match.file_name}")
            print(f"Path: {match.file_path}")
            print(f"Page: {match.page_number}")
            print(f"Position: {match.match_position}")
            print(f"Match Type: {match.match_type}")
            print(f'Snippet: "{match.snippet}"')
            print("-" * 50)


def main() -> None:
    """Simple CLI entry point for local folder-based PDF name search."""

    print("PDF Name Search CLI")
    print("-" * 50)
    folder_path = input("Folder path: ").strip()
    names_input = input("Names (comma-separated): ").strip()

    semantic_fallback_input = input("Enable semantic fallback? (y/N): ").strip().lower()
    semantic_enabled = semantic_fallback_input in {"y", "yes"}

    semantic_mode = "fallback"
    if semantic_enabled:
        semantic_all_input = input(
            "Run semantic mode for all names (not just fallback)? (y/N): "
        ).strip().lower()
        if semantic_all_input in {"y", "yes"}:
            semantic_mode = "always"

    try:
        outcome = run_name_search(
            folder_path=folder_path,
            raw_names=names_input,
            enable_semantic_fallback=semantic_enabled,
            semantic_mode=semantic_mode,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    _print_cli_results(outcome)
    csv_path = export_results_to_csv(outcome.results, "results.csv")
    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
