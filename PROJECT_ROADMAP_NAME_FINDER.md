# 1. Project Title

## Local PDF Name Finder for `rag-document-qa`

# 2. Project Goal

Extend the existing Streamlit app into a local, folder-based PDF name verification tool for macOS while preserving the current Document Q&A workflow.  
Exact text search is the primary search mechanism for Phase 1.

# 3. Scope

In scope (Phase 1):
- Keep existing Q&A workflow unchanged and available.
- Add a new UI workflow for local folder-based exact name search.
- Accept one folder path and one name input.
- Recursively scan the folder for PDFs.
- Show page-level match results directly on the same UI page.

Out of scope (Phase 1):
- Multi-name search in one run.
- CSV export.
- Semantic/RAG/Ollama-assisted matching.
- Non-PDF file formats.

# 4. Phase 1 MVP

Deliver a Streamlit workflow where the user:
1. Enters a local folder path.
2. Enters a single name.
3. Clicks a search button.
4. Sees either:
- Match results (file, full path, page, match position/index, snippet), or
- A clear not-found message.

All input and output must happen on the same page.

# 5. Phase 1 Functional Requirements

- The app must keep current Document Q&A functionality intact.
- Add a dedicated workflow mode (for example, `Local PDF Name Finder`).
- Inputs required:
- `Folder Path` (local absolute or valid relative path).
- `Name` (single person name string).
- Perform recursive directory scanning under the provided folder.
- Process only `.pdf` files in Phase 1.
- Extract text page-by-page.
- Run case-insensitive exact matching with safe escaping for user input.
- Return every exact occurrence found across pages/files.
- For each match, return:
- File name.
- Full file path.
- Page number.
- Match position/index in page text.
- Snippet/context around the match.
- If no match exists in scanned PDFs, show a clear `Not Found` result in UI.
- Unreadable/corrupted/password-protected PDFs must be skipped with warnings; app must not crash.

# 6. Phase 1 UI Requirements

- Add a workflow selector entry for the new feature.
- Render both input controls and results on the same page.
- Required controls:
- Text input for folder path.
- Text input for name.
- Search button.
- Validate empty/invalid inputs and show clear inline errors.
- During scan, show progress/spinner.
- Results area should display:
- Found/Not Found status.
- If found: list/table of matches with file, full path, page number, position/index, snippet.
- If not found: explicit user-friendly message.
- If files were skipped: show warning list without blocking successful results.

# 7. Phase 1 Technical Requirements

- Reuse existing Streamlit app entrypoint (`src/app.py`) and keep Q&A code path unchanged.
- Implement exact-match search as primary logic in a focused module (for example `src/name_finder.py` or split helpers).
- Use recursive file discovery (`os.walk` or equivalent) for PDFs.
- Use robust PDF extraction (`PyPDF2`) with exception handling at file/page level.
- Use regex-based exact matching with:
- Case-insensitive search.
- Escaped user-provided name text.
- Word-boundary-safe matching to avoid partial-name false positives.
- Capture match start/end indices from page text.
- Build snippet using a fixed context window around each match.
- Return structured result objects for simple UI rendering and testing.
- Add/extend tests for:
- Path validation.
- Recursive PDF discovery.
- Exact match behavior.
- Position/index capture.
- Not-found behavior.
- Corrupted/unreadable file handling.

# 8. Phase 1 Acceptance Criteria

- Existing Document Q&A workflow still runs as before.
- User can run search with folder path + single name from UI.
- App scans nested folders recursively for PDFs.
- Results include file name, full path, page number, match position/index, and snippet.
- If no matches, UI clearly shows `Not Found`.
- Corrupted/unreadable files do not crash the app.
- All results render on the same page where input is entered.
- Basic automated tests pass for Phase 1 exact-search functionality.

# 9. Future Phases

- Multi-name search in a single run.
- CSV export of match results.
- Better extraction and error handling (OCR fallback, improved encrypted-file handling, richer diagnostics).
- Semantic fallback using local Ollama models when exact search returns nothing.
- Deeper integration with broader RAG/QA workflow (cross-link from Q&A context to name-finder evidence).

# 10. Risks / Edge Cases

- Corrupted or password-protected PDFs.
- PDFs with scanned images and no extractable text.
- Very large folders causing long scan times.
- Very large pages producing oversized snippets.
- False positives from weak boundaries if regex is not strict enough.
- Duplicate matches due to normalization differences.
- Invalid folder paths or missing permissions on macOS.

# 11. Suggested File/Module Structure

- `PROJECT_ROADMAP_NAME_FINDER.md` (this roadmap)
- `src/app.py`
- Add new UI workflow section for `Local PDF Name Finder`.
- `src/name_finder.py`
- Core search logic (scan, extract, exact match, snippet, result model).
- `tests/test_name_finder.py`
- Unit tests for recursive scanning, exact matching, indices, and error handling.
- `README.md`
- Short usage section for the new workflow.

# 12. Recommended Development Order

1. Add workflow toggle and same-page UI inputs in `src/app.py`.
2. Implement recursive PDF discovery and input validation.
3. Implement page-level PDF text extraction with safe error handling.
4. Implement exact regex matching with match indices and snippet builder.
5. Wire results to UI rendering (found vs not found, skipped file warnings).
6. Add/expand tests for MVP behavior and edge cases.
7. Update README with quick usage instructions.

# Immediate Next Step

Build a UI page where the user enters a folder path and a name, runs the search, and sees matching file/page/position/snippet results or a not found message directly on the same page.
