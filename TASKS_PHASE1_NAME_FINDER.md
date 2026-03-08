# Phase 1 MVP Task Checklist: Local PDF Name Finder

## 1) Phase 1 Guardrails
- [ ] Keep existing `Document Q&A` workflow fully working.
- [ ] Keep Phase 1 limited to exact-match PDF search only.
- [ ] Do not add semantic/RAG/Ollama behavior in Phase 1 implementation.
- [ ] Do not add multi-name search or CSV export in Phase 1.

## 2) UI Workflow in `src/app.py`
- [ ] Add/select a workflow mode for `Local PDF Name Finder`.
- [ ] Add `Folder Path` input field.
- [ ] Add `Name` input field (single name only).
- [ ] Add a `Search` button to trigger execution.
- [ ] Render inputs and results on the same page.
- [ ] Add inline validation for empty folder path or empty name.
- [ ] Show progress/spinner while scanning and searching.

## 3) Core Search Logic in `src/name_finder.py`
- [ ] Implement folder path validation (`exists`, `is_dir`).
- [ ] Implement recursive PDF discovery (`.pdf`, case-insensitive).
- [ ] Extract text page-by-page with `PyPDF2`.
- [ ] Skip unreadable/corrupted/password-protected files without crashing.
- [ ] Implement exact, case-insensitive, escaped regex matching.
- [ ] Enforce boundary-safe matching to avoid partial-name false positives.
- [ ] Capture match position/index (start/end) in page text.
- [ ] Build snippet/context around each match with a fixed window.
- [ ] Return structured page-level results for UI rendering.

## 4) UI Result Rendering (Same Page)
- [ ] Show clear `Found` status when at least one match exists.
- [ ] Display per-match fields:
- [ ] file name
- [ ] full file path
- [ ] page number
- [ ] match position/index
- [ ] snippet/context
- [ ] Show clear `Not Found` message when no matches exist.
- [ ] Show skipped-file warnings (non-blocking).

## 5) Test Coverage in `tests/test_name_finder.py`
- [ ] Test folder path validation (invalid path errors).
- [ ] Test recursive PDF discovery in nested folders.
- [ ] Test exact-match behavior (case-insensitive and boundary-safe).
- [ ] Test position/index capture for matches.
- [ ] Test snippet generation around match location.
- [ ] Test not-found behavior.
- [ ] Test corrupted/unreadable file handling (skip, no crash).

## 6) Stability and Review
- [ ] Run targeted tests for name-finder module.
- [ ] Run full test suite and confirm no regression in existing Q&A path.
- [ ] Manually verify UI flow end-to-end on local macOS sample PDFs.
- [ ] Confirm same-page rendering for input + result.

## 7) Documentation Updates
- [ ] Update `README.md` with Phase 1 usage steps:
- [ ] enter folder path
- [ ] enter name
- [ ] run search
- [ ] interpret found/not-found results
- [ ] Note Phase 1 limitation: exact PDF search only.

## Definition of Done (Phase 1)
- [ ] User can enter folder path + single name in UI.
- [ ] App scans PDFs recursively and returns page-level exact matches.
- [ ] Each result shows file, full path, page, position/index, snippet.
- [ ] `Not Found` message appears clearly when no match exists.
- [ ] Corrupted/unreadable PDFs are skipped safely with warning output.
- [ ] Existing Document Q&A workflow remains functional.
