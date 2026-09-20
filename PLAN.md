# watfile — PLAN

**Goal**: CLI tool that classifies files into categories using a decision-model backend and sorts them into folders.

**Stack**: Python 3.12+, uv, `typesafe-sdk` (Jev, primary), `laya`/MLX-OpenAI-HTTP (local option, later), `liteparse` (Python API, PDF→text).

## Architecture

```
watfile/
├── pyproject.toml            # uv project
├── src/watfile/
│   ├── __init__.py
│   ├── cli.py                # argparse: files/folder, -r, -d/-c, --backend
│   ├── extract.py            # text extraction: plain read for .txt/.md, liteparse for .pdf
│   ├── classifier/
│   │   ├── base.py           # Classifier ABC: classify(text, categories) -> Verdict(category, confidence)
│   │   ├── jev.py            # TypeSafe Choice implementation
│   │   └── laya.py           # (stub later) local MLX via OpenAI-compatible HTTP
│   └── sorter.py             # move/copy files into category folders, dry-run, collision handling
└── tests/
    ├── test_extract.py
    ├── test_classifier.py    # unit with mocked backend
    └── test_sorter.py
```

## Steps

| id | step | priority | status |
|----|------|----------|--------|
| 10 | uv project scaffold + deps (typesafe-sdk 0.7.0, liteparse 2.14.6) | 10 | done |
| 20 | extract.py: text/PDF extraction via liteparse | 9 | done |
| 30 | classifier/base.py + jev.py (Choice primitive) | 10 | done |
| 40 | sorter.py: category folder resolution, move/copy, dry-run | 9 | done |
| 50 | cli.py: file list/folder/-r, -c list or -d folder scan, --backend | 10 | done |
| 60 | unit tests (extract/classifier/sorter), mocked Jev — 9 green | 8 | done |
| 70 | integration smoke test — written, skips w/o TYPESAFE_API_KEY (no key in integration.env) | 6 | blocked: need API key |
| 80 | laya backend stub (local MLX OpenAI-compatible HTTP) | 4 | later |
| 90 | batching N files per call | 4 | later |

## Decisions (from user)

- Python + uv (not Rust) — TypeSafe/laya/liteparse all have native Python/TS SDKs.
- Jev Choice question for classification; abstraction trait allows laya/other backends.
- liteparse Python API for PDFs.
- One file per API call in v1; batching (25/50/100) deferred but trait shaped for it.
- CLI: `watfile <files...|folder> [-r] (-c cat1,cat2 | -d <folder with category subfolders>)`.

## Learnings (project-local)

- **typesafe-sdk**: `system_one(state, {"k": Choice(instructions, criteria={cat: None,...})})` → `response.choices["k"].choice/.confidence/.probabilities`. Model `jev-latest`, auth via `TYPESAFE_API_KEY` env. ~1 call/s sustained; bursts of 4+ in tight pytest loops fail — run live tests serially.
- **Config layering**: env > ./.env (gitignored) > ~/.config/watfile/config.toml. Normalize TOML friendly keys (`api_key`) to env names (`TYPESAFE_API_KEY`) explicitly. If a test writes `os.environ` directly, clear in teardown or downstream live tests 401 on the leaked fake key.
- **liteparse**: `LiteParse(ocr_enabled=False, max_pages=2)` — arXiv PDFs 1.6–7.3s → 9–30ms per file, title/abstract retained. OCR only needed for scanned/image PDFs (which we skip: no text → skipped file).
- **PyPI**: name `watfile` free (2026-09). Publish via `pypa/gh-action-pypi-publish@release/v1`, `id-token: write`, `release` environment — trusted publishing, no stored token. README documents `uvx watfile` / `uv tool install watfile`.
