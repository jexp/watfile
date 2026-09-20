# watfile

Classify files with a decision-making AI model and sort them into category folders.

watfile sends each document's text (title/abstract-grade extract) to a
**TypeSafe AI Jev** (System One) [Choice](https://docs.typesafe.ai/primitives/choice)
question, gets back a typed answer with a selected category, per-category
probabilities and confidence, then moves the file into the matching folder.
A `Classifier` abstraction keeps the backend pluggable — local MLX (laya) and
other backends slot in later.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

### From PyPI (once published)

```sh
# one-off run, no install
uvx watfile --help

# persistent CLI on your PATH
uv tool install watfile
watfile --help
```

### From source

```sh
git clone <repo> && cd watfile
uv sync            # create venv + install deps (typesafe-sdk, liteparse)
uv run watfile --help

# or install the local checkout as a tool
uv tool install --from . watfile
```

### Configuration

watfile resolves its TypeSafe API key (create one at <https://console.typesafe.ai/>)
with this precedence — first match wins:

1. `TYPESAFE_API_KEY` environment variable
2. `.env` file in the current directory (gitignored; `TYPESAFE_API_KEY=...`)
3. `~/.config/watfile/config.toml` (`api_key = "..."`, also `base_url`, `model`;
   `$WATFILE_CONFIG` or `$XDG_CONFIG_HOME` can relocate it)

```sh
export TYPESAFE_API_KEY=...        # option 1
echo 'TYPESAFE_API_KEY=...' > .env # option 2
cat > ~/.config/watfile/config.toml <<'EOF'   # option 3
api_key = "..."
EOF
```

## Usage

Point watfile at files or folders, and either give a comma-separated category
list (`-c`) or a target folder whose subfolders are the categories (`-d`):

```sh
# explicit categories, files moved into ./sorted/<category>/
uv run watfile ~/Downloads/invoice.pdf -c invoice,donation,apartment

# folder input, recursive; categories = existing subfolders of -d
mkdir -p ~/docs/{invoice,donation,apartment}
uv run watfile ~/Downloads -r -d ~/docs

# preview without touching anything
uv run watfile ~/Downloads -r -d ~/docs -n

# actually move the files (default is symlinking into the category folders)
uv run watfile ~/Downloads -r -d ~/docs -m

# copy instead
uv run watfile ~/Downloads -r -d ~/docs --copy

# custom output root with -c
uv run watfile *.pdf -c computerscience,biology -o ~/sorted
```

Output per file:

```
bill.pdf: invoice (conf 0.94) -> symlink to ~/docs/invoice/bill.pdf
```

Files that can't be classified (unsupported extension, no extractable text) are
skipped with a warning; name collisions get a `_1`, `_2`… suffix.

### Supported inputs

- **Text formats** (read directly): `.txt .md .markdown .rst .log .csv .json`
- **PDF** (via [liteparse](https://github.com/run-llama/liteparse)): only the
  first 2 pages are parsed, OCR disabled — enough for classification, ~1000x
  faster than a full parse. Scanned/image-only PDFs are skipped.

### Backends

- **`jev` (default)** — [TypeSafe AI](https://docs.typesafe.ai) Jev, cloud API.
  Needs `TYPESAFE_API_KEY`. Highest accuracy (4/4 on the arXiv fixtures).
- **`laya`** — local [laya-mlx](https://github.com/mizorewww/laya-mlx) typed
  decision model on Apple Silicon (MLX, ~13ms/decision, fully offline after a
  one-time ~1GB checkpoint download). No API key needed. Same question shape
  as Jev. Because laya's context window is small (512–1024 tokens), the backend
  defaults to **adaptive multi-chunk classification**: the extract is split
  into ~200-token chunks; chunk 1 decides if its probability is decisive
  (≥0.5), otherwise further chunks are classified and probabilities aggregated
  until the decision is decisive (max 10). On the arXiv fixtures: 3/4 (Jev 4/4).

```sh
watfile ~/Downloads -r -d ~/docs --backend laya
# pick a checkpoint via config or env:
#   ~/.config/watfile/config.toml -> laya_model = "aac6fef/laya-multilingual-mlx"
#   or LAYA_MODEL=aac6fef/laya-mlx
```

### Options

```
usage: watfile [-h] [-r] (-c CATEGORIES | -d DIRECTORY) [-o OUTPUT]
               [--backend {jev,laya}] [--batch N]
               [--chunk-tokens CHUNK_TOKENS] [--chunks N] [-n]
               [-m | --copy | --symlink]
               inputs [inputs ...]

options:
  -h, --help            show this help message and exit
  -r, --recursive       recurse into folder inputs
  -c CATEGORIES         comma-separated categories
  -d DIRECTORY          target folder whose existing subfolders are the categories
  -o OUTPUT             output root for sorted files (default: same as -d, or ./sorted with -c)
  --backend {jev,laya}  classifier backend (default: jev)
  --batch N             classify N files per API call (jev: documents
                        packed into one system_one call, ~256 tokens each,
                        ~100 docs per 30k-token window)
  --chunk-tokens N      per-document token budget (default: backend-specific)
  --chunks N            split each document into N chunks, aggregate
                        probabilities; 0 = adaptive. Default: 0 for laya,
                        1 for jev
  -n, --dry-run         print decisions without placing files
  -m, --move            move files into the category folder (default: symlink)
  --copy                copy files instead of symlinking
  --symlink             create symlinks in category folders (default)
```

Batching example:

```sh
# classify 100 files at ~4 API calls instead of 100
uv run watfile ~/Downloads -r -d ~/docs --batch 100
```

## Development

```sh
uv sync
uv run pytest              # unit tests; live API tests skip without TYPESAFE_API_KEY
```

`tests/fixture/` contains 4 real arXiv PDFs with ground-truth categories
(derived from their arXiv subject tags) used by the integration tests.

## Roadmap

- `laya` local backend (MLX via OpenAI-compatible HTTP)
- batching: classify 25/50/100 files in a single API call
