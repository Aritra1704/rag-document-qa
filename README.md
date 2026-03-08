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

3. (Optional, for Q&A answers via Claude) set your API key:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

4. Run the Streamlit app:

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

### 2) PDF Name Search (new behavior)

Select `Workflow -> PDF Name Search`, then provide on the same page:

- `Folder Path` (example: `/Users/aritra/Documents/pdfs`)
- `Name to Search` (example: `John Smith`)
- Click `Search PDFs`

UI output includes:

- total PDFs found
- total matches found
- searched name
- file name
- full file path
- page number
- match position/index
- snippet/context around each match
- `match_type=exact`
- skipped file warnings (corrupted/password-protected/no text)
- clear not-found message when there are no matches

Optional diagnostics:

- Enable `Show extraction debug details` to inspect per-file/per-page extractor attempts
- Includes attempted extractor order, successful extractor, char counts, whitespace-only status, preview text, and errors
- Includes environment diagnostics (`sys.executable`, `sys.version`, extractor importability, and `pdftotext` command availability)
- Includes a quick one-file diagnostic action (first 3 pages) for fast troubleshooting
- Detailed page-level diagnostics are capped in the UI for responsiveness, while summary metrics still cover all scanned files/pages

## Exact Search Behavior (Primary)

- Recursive folder scan of `*.pdf` only
- Case-insensitive regex matching
- Names safely escaped for regex
- Phase 1 UI input is a single name per search
- Page-by-page matching
- Text extraction fallback order: `PyPDF2 -> pypdf -> pdfplumber -> PyMuPDF(fitz) -> pdftotext` (when available)
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

## CLI Entry Point

Run:

```bash
python3 -m src.name_finder
```

CLI prompts:

1. folder path
2. comma-separated names
3. semantic fallback (yes/no)
4. optional semantic-all mode

Then prints grouped matches and saves `results.csv`.

## Updated File Structure

Key files for this feature:

- `src/app.py` (workflow switch, PDF Name Search UI, Local Ollama RAG UI)
- `src/ollama_rag.py` (local Ollama diagnostics, embeddings, and chat helpers)
- `src/name_finder.py` (folder scan, PDF page extraction, exact/semantic search, CSV export, CLI)
- `tests/test_name_finder.py` (tests for name-search utilities and extraction debug behavior)
