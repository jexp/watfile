"""watfile CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .classifier.base import Classifier, Verdict
from .classifier.jev import JevClassifier
from .classifier.laya import LayaClassifier
from .classifier.multi import MultiChunkClassifier, chunk_text_tokens, count_tokens
from .config import apply_config, load_config
from .extract import UnsupportedFileTypeError, extract_text
from .sorter import PLACEMENT_COPY, PLACEMENT_MOVE, PLACEMENT_SYMLINK, place_file

#: Jev context window (tokens) with headroom for state wrapper + questions.
JEV_WINDOW_TOKENS = 32_000 - 2_000

#: Per-document token budget for Jev batching. Measured: 64 tokens (title +
#: first sentence) already give 4/4 correct with conf 1.00 on the arXiv
#: fixtures AND on same-genre German documents (invoice/donation/contract);
#: larger budgets don't improve accuracy. 256 tokens is a safe margin that
#: covers title + abstract/letterhead + first table, and still packs ~100
#: documents into the 30k-token batch window. Override with --chunk-tokens.
JEV_DOC_TOKENS = 256


def _collect_files(inputs: Sequence[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(p for p in sorted(path.glob(pattern)) if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: not found, skipped: {path}", file=sys.stderr)
    # de-duplicate, keep order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(f)
    return unique


def _categories_from_dir(target: Path) -> list[str]:
    if not target.is_dir():
        raise SystemExit(f"target folder does not exist: {target}")
    subdirs = [p.name for p in sorted(target.iterdir()) if p.is_dir()]
    if not subdirs:
        raise SystemExit(
            f"target folder {target} has no subfolders to use as categories; "
            "create one folder per category or pass -c instead"
        )
    return subdirs


def _parse_categories_arg(raw: str) -> list[str]:
    cats = [c.strip() for c in raw.split(",") if c.strip()]
    if len(cats) < 2:
        raise SystemExit("need at least 2 categories for -c")
    return cats


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """First *max_tokens* tokens of *text* (whitespace-clean cut)."""
    parts = chunk_text_tokens(text, max_tokens)
    return parts[0] if parts else ""


def _pack_batches(
    texts: dict[int, str], doc_tokens: int, window_tokens: int, max_files: int
) -> list[list[int]]:
    """Greedily pack file indices into batches that fit *window_tokens* in total.

    texts maps file-index -> trimmed text. Returns lists of file indices.
    A single oversized document still gets its own batch (system_one will
    decide what to do with it).
    """
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for idx, text in texts.items():
        cost = count_tokens(text) + 20  # id wrapper overhead
        if current and (used + cost > window_tokens or len(current) >= max_files):
            batches.append(current)
            current, used = [], 0
        current.append(idx)
        used += cost
    if current:
        batches.append(current)
    return batches


def _build_classifier(name: str, config) -> Classifier:
    if name == "jev":
        if not config.api_key:
            raise SystemExit(
                "no TYPESAFE_API_KEY found.\n"
                "Set the environment variable, or create a gitignored .env file with\n"
                "TYPESAFE_API_KEY=..., or ~/.config/watfile/config.toml with:\n"
                'api_key = "..."\n'
                "Get a key at https://console.typesafe.ai/"
            )
        apply_config(config)
        model = config.model or "jev-latest"
        return JevClassifier(model=model)
    if name == "laya":
        return LayaClassifier(model=config.laya_model)
    raise SystemExit(f"unknown backend: {name}")


def _classify_one(classifier: Classifier, path: Path, categories: Sequence[str]) -> Verdict | None:
    try:
        text = extract_text(path)
    except UnsupportedFileTypeError as exc:
        print(f"  skipped ({exc})", file=sys.stderr)
        return None
    if not text.strip():
        print("  skipped (no extractable text)", file=sys.stderr)
        return None
    return classifier.classify(text, categories)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="watfile",
        description="Classify files with a decision model and sort them into category folders.",
    )
    parser.add_argument("inputs", nargs="+", help="files and/or folders to process")
    parser.add_argument("-r", "--recursive", action="store_true", help="recurse into folder inputs")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("-c", "--categories", help="comma-separated categories, e.g. invoice,donation,apartment")
    target.add_argument("-d", "--directory", help="target folder whose existing subfolders are the categories")
    parser.add_argument("-o", "--output", help="output root for sorted files (default: same as -d, or ./sorted with -c)")
    parser.add_argument("--backend", default="jev", choices=["jev", "laya"], help="classifier backend (default: jev)")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,  # None = auto (on for jev, off for laya)
        metavar="N",
        help="cap files per API call (default: automatic — batches everything that "
        "fits the context window for jev; no batching for laya)",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="disable batching (one API call per file; useful for debugging)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="per-document token budget when batching or chunking (default: auto)",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=-1,  # backend-specific default
        metavar="N",
        help="split each document into N token-sized chunks and aggregate probabilities; 0 = adaptive (extend chunk by chunk until the decision is decisive). Default: 0 for laya, 1 for jev",
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="print decisions without placing files")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("-m", "--move", action="store_true", help="move files into the category folder (default: symlink)")
    placement.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    placement.add_argument("--symlink", action="store_true", help="create symlinks in category folders (default)")
    args = parser.parse_args(argv)

    if args.categories:
        categories = _parse_categories_arg(args.categories)
        target_root = Path(args.output) if args.output else Path("sorted")
    else:
        target_root = Path(args.directory).expanduser()
        categories = _categories_from_dir(target_root)
        if args.output:
            target_root = Path(args.output).expanduser()

    files = _collect_files(args.inputs, args.recursive)
    if not files:
        print("no files to process", file=sys.stderr)
        return 1

    print(f"categories: {', '.join(categories)}")
    print(f"files: {len(files)}  backend: {args.backend}  target: {target_root}"
          + ("  (dry-run)" if args.dry_run else ""))

    classifier = _build_classifier(args.backend, load_config())
    chunks = args.chunks if args.chunks >= 0 else (0 if args.backend == "laya" else 1)
    if chunks != 1:
        classifier = MultiChunkClassifier(classifier, chunks=chunks)

    placement = (
        PLACEMENT_MOVE if args.move else PLACEMENT_COPY if args.copy else PLACEMENT_SYMLINK
    )
    actions = {
        PLACEMENT_MOVE: "move",
        PLACEMENT_COPY: "copy",
        PLACEMENT_SYMLINK: "symlink",
    }[placement]

    def _place(path: Path, verdict) -> None:
        result = place_file(path, verdict.category, target_root, dry_run=args.dry_run, placement=placement)
        prefix = "would " if args.dry_run else ""
        print(f"{verdict.category} (conf {verdict.confidence:.2f}) -> {prefix}{actions} to {result.destination}")

    failures = 0
    # batching is automatic for jev (strictly better: fewer calls, same or
    # better accuracy per measurement), off for laya (local inference is
    # already fast), disabled by --no-batch or when multi-chunking is active
    # (batching × chunking would multiply calls — composition needs design).
    batch_on = (
        not args.no_batch
        and args.backend == "jev"
        and not isinstance(classifier, MultiChunkClassifier)
    )
    if batch_on:
        batch_cap = args.batch if args.batch is not None else 10_000  # effectively "window only"
        doc_tokens = args.chunk_tokens or JEV_DOC_TOKENS
        texts: dict[int, str] = {}
        for i, path in enumerate(files):
            try:
                text = extract_text(path)
            except UnsupportedFileTypeError as exc:
                print(f"{path.name}: skipped ({exc})", file=sys.stderr)
                failures += 1
                continue
            if not text.strip():
                print(f"{path.name}: skipped (no extractable text)", file=sys.stderr)
                failures += 1
                continue
            texts[i] = _trim_to_tokens(text, doc_tokens)
        for batch in _pack_batches(texts, doc_tokens, JEV_WINDOW_TOKENS, batch_cap):
            print(f"batch of {len(batch)} file(s)...", flush=True)
            verdicts = classifier.classify_batch([texts[i] for i in batch], categories)
            for i, verdict in zip(batch, verdicts):
                print(f"{files[i].name}: ", end="")
                _place(files[i], verdict)
    else:
        for path in files:
            print(f"{path.name}: ", end="", flush=True)
            verdict = _classify_one(classifier, path, categories)
            if verdict is None:
                failures += 1
                continue
            _place(path, verdict)

    return 1 if failures == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
