# rag-document-qa

Streamlit app with two local workflows:

1. `Document Q&A` (existing RAG chat flow)
2. `PDF Name Search` (new folder-based name verification utility)

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
- Detailed page-level diagnostics are capped in the UI for responsiveness, while summary metrics still cover all scanned files/pages

## Exact Search Behavior (Primary)

- Recursive folder scan of `*.pdf` only
- Case-insensitive regex matching
- Names safely escaped for regex
- Phase 1 UI input is a single name per search
- Page-by-page matching
- Match position/index is captured from page text
- Snippet extraction around each match (not full page dump)
- Multiple matches on same page are returned, with duplicate suppression

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

- `src/app.py` (added PDF Name Search workflow mode in Streamlit)
- `src/name_finder.py` (folder scan, PDF page extraction, exact/semantic search, CSV export, CLI)
- `tests/test_name_finder.py` (new tests for name-search utilities)
