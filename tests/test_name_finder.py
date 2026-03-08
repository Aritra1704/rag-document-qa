"""Tests for folder-based PDF name finder utilities."""

import csv
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import name_finder  # noqa: E402
from name_finder import (  # noqa: E402
    ExtractorOpenDebug,
    NameMatch,
    PageRecord,
    collect_pdf_pages,
    discover_pdf_files,
    export_results_to_csv,
    find_exact_name_matches,
    parse_names,
    run_name_search,
    summarize_extraction_debug,
)


def test_parse_names_dedupes_and_normalizes():
    names = parse_names(" John Smith, jane doe,JOHN   SMITH, , Jane   Doe ")
    assert names == ["John Smith", "jane doe"]


def test_discover_pdf_files_recursive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()

    (tmp_path / "one.pdf").write_text("ignored content", encoding="utf-8")
    (nested / "two.PDF").write_text("ignored content", encoding="utf-8")
    (nested / "notes.txt").write_text("not a pdf", encoding="utf-8")

    discovered = discover_pdf_files(tmp_path)
    discovered_names = sorted(path.name for path in discovered)
    assert discovered_names == ["one.pdf", "two.PDF"]


def test_collect_pdf_pages_uses_fallback_extractor(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    primary = name_finder._PdfExtractor(
        name="PyPDF2",
        page_count=1,
        extract_page_text=lambda _: ("   ", None),
        close=lambda: None,
    )
    fallback = name_finder._PdfExtractor(
        name="pypdf",
        page_count=1,
        extract_page_text=lambda _: ("Agreement signed by John Smith.", None),
        close=lambda: None,
    )

    monkeypatch.setattr(
        name_finder,
        "_build_pdf_extractors",
        lambda _: (
            {"PyPDF2": primary, "pypdf": fallback},
            {
                "PyPDF2": ExtractorOpenDebug("PyPDF2", True, True, True, None),
                "pypdf": ExtractorOpenDebug("pypdf", True, True, True, None),
                "pdfplumber": ExtractorOpenDebug("pdfplumber", False, False, False, "import failed"),
                "pymupdf": ExtractorOpenDebug("pymupdf", False, False, False, "import failed"),
                "pdftotext": ExtractorOpenDebug("pdftotext", False, False, False, "command not found"),
            },
        ),
    )

    pages, skipped = collect_pdf_pages([pdf_path])
    assert len(pages) == 1
    assert pages[0].file_name == "sample.pdf"
    assert pages[0].page_number == 1
    assert "John Smith" in pages[0].text
    assert skipped == []


def test_collect_pdf_pages_marks_file_only_after_all_extractors_fail(monkeypatch, tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    extractors = [
        name_finder._PdfExtractor(
            name="PyPDF2",
            page_count=1,
            extract_page_text=lambda _: ("   ", None),
            close=lambda: None,
        ),
        name_finder._PdfExtractor(
            name="pypdf",
            page_count=1,
            extract_page_text=lambda _: ("\n\t", None),
            close=lambda: None,
        ),
        name_finder._PdfExtractor(
            name="pdfplumber",
            page_count=1,
            extract_page_text=lambda _: ("", None),
            close=lambda: None,
        ),
    ]

    monkeypatch.setattr(
        name_finder,
        "_build_pdf_extractors",
        lambda _: (
            {extractor.name: extractor for extractor in extractors},
            {
                "PyPDF2": ExtractorOpenDebug("PyPDF2", True, True, True, None),
                "pypdf": ExtractorOpenDebug("pypdf", True, True, True, None),
                "pdfplumber": ExtractorOpenDebug("pdfplumber", True, True, True, None),
                "pymupdf": ExtractorOpenDebug("pymupdf", False, False, False, "import failed"),
                "pdftotext": ExtractorOpenDebug("pdftotext", False, False, False, "command not found"),
            },
        ),
    )

    pages, skipped = collect_pdf_pages([pdf_path])
    assert pages == []
    assert len(skipped) == 1
    assert "no extractable text after PyPDF2, pypdf, pdfplumber, pymupdf, and pdftotext" in skipped[0]


def test_collect_pdf_pages_debug_and_summary(monkeypatch, tmp_path):
    pdf_path = tmp_path / "debug.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")

    primary = name_finder._PdfExtractor(
        name="PyPDF2",
        page_count=2,
        extract_page_text=lambda index: ("John Smith", None) if index == 0 else ("", None),
        close=lambda: None,
    )

    monkeypatch.setattr(
        name_finder,
        "_build_pdf_extractors",
        lambda _: (
            {"PyPDF2": primary},
            {
                "PyPDF2": ExtractorOpenDebug("PyPDF2", True, True, True, None),
                "pypdf": ExtractorOpenDebug("pypdf", False, False, False, "import failed"),
                "pdfplumber": ExtractorOpenDebug("pdfplumber", False, False, False, "import failed"),
                "pymupdf": ExtractorOpenDebug("pymupdf", False, False, False, "import failed"),
                "pdftotext": ExtractorOpenDebug("pdftotext", False, False, False, "command not found"),
            },
        ),
    )

    pages, skipped, debug_entries = collect_pdf_pages([pdf_path], include_debug=True)
    assert len(pages) == 1
    assert skipped == []
    assert len(debug_entries) == 1
    assert len(debug_entries[0].page_debug) == 2
    assert debug_entries[0].extractor_open_debug[0].extractor_name == "PyPDF2"
    first_page_attempts = debug_entries[0].page_debug[0].attempts
    assert len(first_page_attempts) == 5
    assert first_page_attempts[0].extractor_name == "PyPDF2"
    assert first_page_attempts[0].extraction_attempted is True
    assert first_page_attempts[0].succeeded is True
    assert first_page_attempts[1].extraction_attempted is False
    assert "skipped after winner" in (first_page_attempts[1].error or "")

    summary = summarize_extraction_debug(debug_entries)
    assert summary["pdfs_discovered"] == 1
    assert summary["pdfs_with_extracted_text"] == 1
    assert summary["pdfs_fully_skipped"] == 0
    assert summary["pages_with_text"] == 1
    assert summary["pages_with_no_text"] == 1


def test_find_exact_name_matches_case_insensitive_and_word_boundaries():
    page_records = [
        PageRecord(
            file_name="contract.pdf",
            file_path="/tmp/contract.pdf",
            page_number=12,
            text=(
                "Agreement signed by John Smith on behalf of the company. "
                "Later, john smith approved a revision. "
                "This line mentions John Smithers and should not match."
            ),
        )
    ]

    matches = find_exact_name_matches(page_records, ["John Smith"])
    assert len(matches) == 2
    assert all(match.match_type == "exact" for match in matches)
    assert all("John Smith" in match.snippet for match in matches)
    positions = sorted(match.match_position for match in matches)
    assert positions[0] >= 0
    assert positions[1] > positions[0]


def test_export_results_to_csv_writes_expected_columns(tmp_path):
    results = [
        NameMatch(
            searched_name="John Smith",
            file_name="contract.pdf",
            file_path="/tmp/contract.pdf",
            page_number=3,
            match_position=21,
            snippet="... John Smith ...",
            match_type="exact",
        )
    ]
    output_path = export_results_to_csv(results, tmp_path / "results.csv")

    with output_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "searched_name",
        "file_name",
        "file_path",
        "page_number",
        "snippet",
        "match_type",
    ]
    assert len(rows) == 1
    assert rows[0]["searched_name"] == "John Smith"
    assert rows[0]["match_type"] == "exact"


def test_run_name_search_invalid_folder_raises():
    with pytest.raises(ValueError):
        run_name_search("/path/does/not/exist", "John Smith")


def test_run_name_search_requires_at_least_one_name(tmp_path):
    with pytest.raises(ValueError):
        run_name_search(tmp_path, "  ,  ")
