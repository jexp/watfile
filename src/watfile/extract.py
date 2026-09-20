"""Text extraction from files for classification."""

from pathlib import Path

from liteparse import LiteParse

#: Extensions read directly as text.
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json"}

#: Extensions parsed via liteparse.
PARSED_EXTENSIONS = {".pdf"}

#: Cap extracted text sent to the classifier (chars). Jev works on decision-relevant
#: context; whole documents are unnecessary and slow.
MAX_TEXT_CHARS = 8_000

#: Only parse the first pages of PDFs — title/abstract carry the classification signal,
#: and parsing hundreds of pages wastes seconds per file.
MAX_PDF_PAGES = 2


class UnsupportedFileTypeError(ValueError):
    pass


def extract_text(path: Path) -> str:
    """Return text content of *path*, truncated to MAX_TEXT_CHARS.

    Plain-text formats are read directly; PDFs go through liteparse.
    """
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext in PARSED_EXTENSIONS:
        # OCR off: classification only needs embedded text, and OCR adds 1.5-7s/page.
        # Scanned-image PDFs will come back empty and are skipped by the caller.
        result = LiteParse(ocr_enabled=False, max_pages=MAX_PDF_PAGES, quiet=True).parse(path)
        text = result.text
    else:
        raise UnsupportedFileTypeError(f"unsupported extension: {ext or '<none>'} ({path.name})")
    return text[:MAX_TEXT_CHARS]
