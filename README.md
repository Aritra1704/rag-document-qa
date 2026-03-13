# rag-document-qa

Streamlit app with three local workflows:

1. `Document Q&A` (existing RAG chat flow)
2. `PDF Name Search` (new folder-based name verification utility)
3. `Local Ollama RAG` (local semantic retrieval and QA)

The Q&A flow is preserved. The name-search flow is additive and optimized for exact name verification in local PDFs.

## Python Version

This project works with **Python 3.11**.

## Local Setup (Mac Mini / macOS)

1. Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

OpenCV/Numpy compatibility for card detection:

- This project expects `numpy>=1.26,<2.0` with `opencv-python>=4.10.0`.
- If you see `cv2`/numpy mismatch errors, clean and reinstall inside the venv:

```bash
pip uninstall -y opencv-python opencv-python-headless numpy
pip install -r requirements.txt
```

3. Install Tesseract (required for OCR fallback on scanned/image PDFs):

```bash
brew install tesseract
```

4. Configure PostgreSQL for PDF Name Search persistence:

```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/rag_document_qa"
```

Initialize tables/indexes (one-time):

```bash
psql "$DATABASE_URL" -f db/postgres/000_create_project_schema.sql
psql "$DATABASE_URL" -f db/postgres/001_name_search_schema.sql
psql "$DATABASE_URL" -f db/postgres/002_name_search_indexes.sql
```

5. (Optional, for Q&A answers via Claude) set your API key:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

6. Run the Streamlit app:

```bash
streamlit run src/app.py
```

## Local Ollama Setup (for Local Ollama RAG workflow)

1. Start Ollama locally (macOS):

```bash
ollama serve
```

2. Pull required local models:

```bash
ollama pull nomic-embed-text:latest
ollama pull qwen2.5:7b-instruct
```

3. In the app, choose `Workflow -> Local Ollama RAG` and keep defaults unless needed:

- `Ollama Base URL`: `http://localhost:11434`
- `Embedding Model`: `nomic-embed-text:latest`
- `Chat Model`: `qwen2.5:7b-instruct`

## Workflows

### 1) Document Q&A (existing behavior)

- Upload PDFs/DOCX/TXT from the sidebar.
- Click `Process Documents`.
- Ask questions in chat.

### 2) PDF Name Search (live + stored workflow)

`Workflow -> PDF Name Search` now has two submodes:

- `Live Scan Search`
- `Search Stored Data`

Main controls:

- `Folder Path` (example: `/Users/aritra/Documents/pdfs`)
- `PostgreSQL DATABASE_URL (optional override)` (if empty, app uses `DATABASE_URL` or `PG*` env vars)
- `Name to Search`
- `Test mode` (default ON)
- `Max files to process in test mode` (default `1`)
- `Max pages per file in test mode` (default `1`)
- `Start page` (default `3`)
- `End page` (default `3` in test mode)
- `Test file` dropdown (auto-first file or select one file)
- `Replace existing records for selected test page` (default ON)
- `Enable OCR fallback`
- `OCR timeout per page (seconds)`
- `Overall timeout (seconds, 0 = no timeout)`

Buttons:

- `Search PDFs`: live progressive scan + exact matching + storage update
- `Process & Store`: process folder pages and store parsed records for future fast queries
- `Run 1-page DB test`: run controlled test-mode pipeline for one file/page
- `Stop Scan`: interrupt current run and keep partial results
- `Search Stored Data`: query previously stored structured records without rerunning OCR

Live UI output includes:

- progressive status (file index/total, current file, page number, stage)
- pages processed, matches found, elapsed time, skipped counts
- partial exact matches as they are found
- final stop reason: completed all files / overall timeout / stopped by user / interrupted
- deterministic result format:
  - searched name
  - file name
  - full path
  - page number
  - match position
  - snippet
  - `match_type` (`exact_text` or `ocr_text`)

Structured parsing + storage:

- page text is parsed into voter-style records and stored locally
- parsed fields:
  - `serial_number`
  - `elector_id`
  - `name`
  - `relative_name`
  - `relative_type`
  - `house_number`
  - `age`
  - `gender`
  - `constituency`
  - `section_name`
  - `file_name`
  - `file_path`
  - `page_number`
  - `extraction_method`
  - `raw_record_text`
- storage tracks statuses like `pending`, `processing`, `processed`, `skipped`, `failed`, `stopped`
- app auto-runs table/index initialization on connect (`CREATE TABLE/INDEX IF NOT EXISTS`)
- dedicated schema used by this project: `rag_document_qa`
- SQLite-to-PostgreSQL data migration is not implemented yet (reprocess PDFs to repopulate Postgres)

Optional diagnostics (Live mode):

- `Show extraction debug details` shows per-file/per-page extractor attempts
- includes OCR fallback debug columns and quick one-file diagnostic (first 3 pages)
- includes raw winner text block per page (first 500 chars)

### 2.2) Offline Batch Ingestion (recommended for full dataset)

For full-folder ingestion, run the dedicated offline ingester instead of keeping Streamlit in a long OCR loop:

```bash
python -m src.ingest_rolls --folder "/absolute/path/to/pdfs"
```

Recommended baseline:

```bash
python -m src.ingest_rolls \
  --folder "/absolute/path/to/pdfs" \
  --workers 2 \
  --start-page 3
```

Useful flags:

- `--workers 1..4`: bounded concurrency (default `2`, capped to `4`)
- `--start-page N` / `--end-page N`: page-range control
- `--max-files N`: limit files for staged runs
- `--disable-ocr-fallback`: disable OCR fallback path
- `--no-resume`: disable checkpoint resume
- `--no-reprocess-changed`: do not force full target-range reprocess for changed files
- `--no-reprocess-failed`: skip retrying failed pages on resume

Batch mode behavior:

- discovers PDFs recursively
- writes `documents` / `pages` checkpoints per file/page
- parses and stores `parsed_records` using current schema
- skips unchanged already-completed files using `file_path + last_modified + file_size`
- resumes interrupted runs from pending pages
- logs:
  - files discovered/processed/skipped/failed
  - pages attempted/skipped
  - records inserted
  - low-confidence record count
  - elapsed time

Low-confidence persistence in batch mode:

- records classified as review-worthy are still inserted
- review marker is preserved in `parsed_records.extraction_method` suffix:
  - `|needs_review`
  - `|partial`

Streamlit integration for batch mode:

- open `Workflow -> PDF Name Search -> Live Scan Search`
- use `Offline Batch Ingestion (Recommended For Full Dataset)` panel
- launch background batch ingestion from UI
- monitor DB-backed progress metrics and current processing file
- keep using one-page test mode for parser tuning

### 2.3) Quick 1-page PostgreSQL test (recommended before full run)

1. Open `Workflow -> PDF Name Search -> Live Scan Search`
2. Keep `Test mode` ON
3. Set:
   - `Max files to process in test mode = 1`
   - `Max pages per file in test mode = 1`
   - `Start page = 3`
   - `End page = 3`
4. Choose a `Test file` (or keep auto-first)
5. Click `Run 1-page DB test`

Turn `Test mode` OFF to restore full-scan behavior using the normal scan controls.

The run still executes the full pipeline on that sample:

- discover file(s)
- text extraction
- OCR fallback (if needed)
- structured parsing
- PostgreSQL insert

After run, UI shows:

- test-run summary (file/page, extraction method, OCR usage, parsed count, insert success count, insert errors)
- PostgreSQL verification section for canonical tables:
  - `documents` row for tested file
  - `pages` row for tested page
  - `parsed_records` rows for tested page
  - row-count checks (`documents=1`, `pages=1`, `parsed_records > 1`)

## Exact Search Behavior (Primary)

- Recursive folder scan of `*.pdf` only
- Case-insensitive regex matching
- Names safely escaped for regex
- Phase 1 UI input is a single name per search
- Page-by-page matching
- Text extraction fallback order: `PyPDF2 -> pypdf -> pdfplumber -> PyMuPDF(fitz) -> pdftotext` (when available)
- OCR fallback: if all extractors return empty/whitespace for a page, OCR is attempted and used for matching
- OCR fallback respects per-page OCR timeout to avoid stalling on one page
- OCR remains fallback only (not first choice)
- Match position/index is captured from page text
- Snippet extraction around each match (not full page dump)
- Multiple matches on same page are returned, with duplicate suppression

### 3) Local Ollama RAG (new behavior)

- Upload PDFs/DOCX/TXT for local embedding and retrieval
- Uses local Ollama embeddings + chat model for semantic QA
- Connectivity diagnostics in UI:
  - endpoint reachability
  - embedding/chat model availability
- Answers are generated from retrieved context with source references

## Deterministic vs LLM-Assisted

- `PDF Name Search` exact matching is deterministic and remains the source of truth for verification.
- `Local Ollama RAG` is LLM-assisted and intended for semantic retrieval, summarization, and question answering.
- Verification should rely on exact-match results, not model guesses.

## Future Enhancements (Post-Phase 1)

- Multi-name search
- CSV export from the UI
- Semantic fallback (for example with local models) when exact search finds nothing
- Deeper integration with broader RAG/QA flow

## CLI Entry Points

Interactive name-search CLI:

```bash
python3 -m src.name_finder
```

CLI prompts:

1. folder path
2. comma-separated names
3. semantic fallback (yes/no)
4. optional semantic-all mode

Then prints grouped matches and saves `results.csv`.

Offline batch ingestion CLI:

```bash
python3 -m src.ingest_rolls --folder "/absolute/path/to/pdfs"
```

## Updated File Structure

Key files for this feature:

- `src/app.py` (workflow switch, PDF Name Search UI, Local Ollama RAG UI)
- `src/ollama_rag.py` (local Ollama diagnostics, embeddings, and chat helpers)
- `src/name_finder.py` (folder scan, PDF page extraction, exact/semantic search, CSV export, CLI)
- `src/ingest_rolls.py` (offline resumable batch ingestion CLI for full datasets)
- `src/name_search_storage.py` (PostgreSQL storage adapter, OCR-text record parser, storage/search helpers)
- `db/postgres/001_name_search_schema.sql` (PostgreSQL table creation)
- `db/postgres/002_name_search_indexes.sql` (PostgreSQL indexes)
- `db/postgres/000_create_project_schema.sql` (schema creation: `rag_document_qa`)
- `tests/test_name_finder.py` (tests for name-search utilities and extraction debug behavior)
