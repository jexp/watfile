import os
from pathlib import Path

from watfile.sorter import PLACEMENT_COPY, PLACEMENT_MOVE, PLACEMENT_SYMLINK, place_file, sanitize_category


def test_sanitize_category() -> None:
    assert sanitize_category("invoices 2026") == "invoices 2026"
    assert sanitize_category("a/b\\c:d") == "a_b_c_d"
    assert sanitize_category("   ") == "uncategorized"
    assert sanitize_category("..hidden..") == "hidden"


def test_place_file_moves_into_category_folder(tmp_path: Path) -> None:
    src = tmp_path / "bill.pdf"
    src.write_text("data")
    root = tmp_path / "sorted"
    result = place_file(src, "invoice", root, placement=PLACEMENT_MOVE)
    assert result.moved
    assert result.destination == root / "invoice" / "bill.pdf"
    assert result.destination.exists()
    assert not src.exists()


def test_place_file_symlinks_by_default(tmp_path: Path) -> None:
    src = tmp_path / "bill.pdf"
    src.write_text("data")
    root = tmp_path / "sorted"
    result = place_file(src, "invoice", root)  # default = symlink
    assert result.moved
    assert result.destination.is_symlink()
    assert result.destination.resolve() == src.resolve()
    assert src.exists()  # original untouched
    assert result.destination.read_text() == "data"  # readable through the link


def test_place_file_symlink_collision_appends_counter(tmp_path: Path) -> None:
    root = tmp_path / "sorted"
    cat_dir = root / "invoice"
    cat_dir.mkdir(parents=True)
    (cat_dir / "bill.pdf").symlink_to(Path("/nonexistent"))  # stale symlink exists
    src = tmp_path / "bill.pdf"
    src.write_text("new")
    result = place_file(src, "invoice", root)
    assert result.destination.name == "bill_1.pdf"


def test_place_file_dry_run_touches_nothing(tmp_path: Path) -> None:
    src = tmp_path / "doc.txt"
    src.write_text("data")
    result = place_file(src, "donation", tmp_path / "sorted", dry_run=True)
    assert not result.moved
    assert src.exists()
    assert not result.destination.exists()


def test_place_file_resolves_name_collision(tmp_path: Path) -> None:
    category_dir = tmp_path / "sorted" / "invoice"
    category_dir.mkdir(parents=True)
    (category_dir / "bill.pdf").write_text("existing")
    src = tmp_path / "bill.pdf"
    src.write_text("new")
    result = place_file(src, "invoice", tmp_path / "sorted", placement=PLACEMENT_MOVE)
    assert result.destination.name == "bill_1.pdf"
    assert result.destination.read_text() == "new"
    assert (category_dir / "bill.pdf").read_text() == "existing"


def test_place_file_copy_keeps_source(tmp_path: Path) -> None:
    src = tmp_path / "keep.txt"
    src.write_text("data")
    result = place_file(src, "apartment", tmp_path / "out", placement=PLACEMENT_COPY)
    assert result.moved and src.exists() and result.destination.exists()
    assert not result.destination.is_symlink()
