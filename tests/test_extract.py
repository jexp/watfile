from pathlib import Path

import pytest

from watfile.extract import MAX_TEXT_CHARS, UnsupportedFileTypeError, extract_text


def test_extract_plain_text(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("hello world")
    assert extract_text(f) == "hello world"


def test_extract_truncates_long_text(tmp_path: Path) -> None:
    f = tmp_path / "big.md"
    f.write_text("x" * (MAX_TEXT_CHARS + 500))
    assert len(extract_text(f)) == MAX_TEXT_CHARS


def test_extract_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(f)
