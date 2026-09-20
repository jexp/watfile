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

```sh
git clone <repo> && cd watfile
uv sync            # create venv + install deps (typesafe-sdk, liteparse)
```

Set your API key (create one at <https://console.typesafe.ai/>):

```sh
export TYPESAFE_API_KEY=ts_...
```

Then run via `uv run watfile ...`, or install the CLI into your path:

```sh
uv tool install --from . watfile
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

# copy instead of move
uv run watfile ~/Downloads -r -d ~/docs --copy

# custom output root with -c
uv run watfile *.pdf -c computerscience,biology -o ~/sorted
```

Output per file:

```
bill.pdf: invoice (conf 0.94) -> moved to ~/docs/invoice/bill.pdf
```

Files that can't be classified (unsupported extension, no extractable text) are
skipped with a warning; name collisions get a `_1`, `_2`… suffix.

### Supported inputs

- **Text formats** (read directly): `.txt .md .markdown .rst .log .csv .json`
- **PDF** (via [liteparse](https://github.com/run-llama/liteparse)): only the
  first 2 pages are parsed, OCR disabled — enough for classification, ~1000x
  faster than a full parse. Scanned/image-only PDFs are skipped.

### Options

```
usage: watfile [-h] [-r] (-c CATEGORIES | -d DIRECTORY) [-o OUTPUT]
               [--backend {jev,laya}] [-n] [--copy]
               inputs [inputs ...]

positional arguments:
  inputs                files and/or folders to process

options:
  -h, --help            show this help message and exit
  -r, --recursive       recurse into folder inputs
  -c CATEGORIES, --categories CATEGORIES
                        comma-separated categories, e.g. invoice,donation,apartment
  -d DIRECTORY, --directory
                        target folder whose existing subfolders are the categories
  -o OUTPUT, --output   output root for sorted files (default: same as -d, or ./sorted with -c)
  --backend {jev,laya}  classifier backend (default: jev)
  -n, --dry-run         print decisions without moving files
  --copy                copy instead of move
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
