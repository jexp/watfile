"""Confidence gating: files below --min-confidence are left in place."""

from pathlib import Path
from unittest.mock import patch

import pytest

from watfile.classifier.base import Classifier, Verdict
from watfile.cli import main


class _FixedClassifier(Classifier):
    def __init__(self, confidence: float, category: str = "invoice", runner_up: float = 0.1) -> None:
        self._verdict = Verdict(
            category=category,
            confidence=confidence,
            probabilities={category: confidence, "other": runner_up},
        )

    def classify(self, text, categories, *, name=None, descriptions=None) -> Verdict:
        return self._verdict

    def classify_batch(self, texts, categories, *, names=None, descriptions=None):
        return [self._verdict for _ in texts]


def _run(tmp_path: Path, confidence: float, extra: list[str] | None = None) -> tuple[int, Path, Path]:
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    (src_dir / "doc.txt").write_text("Invoice #7 amount EUR 99 due next week")
    out = tmp_path / "sorted"
    with patch("watfile.cli._build_classifier", return_value=_FixedClassifier(confidence)):
        rc = main([str(src_dir), "-c", "invoice,other", "-o", str(out)] + (extra or []))
    return rc, src_dir / "doc.txt", out / "invoice" / "doc.txt"


def test_low_confidence_leaves_file_in_place(tmp_path: Path) -> None:
    rc, original, destination = _run(tmp_path, confidence=0.2)
    assert rc == 0
    assert original.exists(), "uncertain file must stay put"
    assert not destination.exists(), "uncertain file must not be placed"


def test_high_confidence_places_file(tmp_path: Path) -> None:
    rc, original, destination = _run(tmp_path, confidence=0.9)
    assert rc == 0
    assert original.exists()  # symlink: original stays
    assert destination.is_symlink()


def test_threshold_is_inclusive_boundary(tmp_path: Path) -> None:
    # exactly at the default threshold (0.5) counts as confident enough
    rc, original, destination = _run(tmp_path, confidence=0.5)
    assert destination.is_symlink()


def test_threshold_zero_disables_gating(tmp_path: Path) -> None:
    rc, original, destination = _run(tmp_path, confidence=0.01, extra=["--min-confidence", "0"])
    assert destination.is_symlink()


def test_custom_threshold(tmp_path: Path) -> None:
    # 0.85 conf with a 0.9 requirement -> skip
    rc, original, destination = _run(tmp_path, confidence=0.85, extra=["--min-confidence", "0.9"])
    assert not destination.exists()


def test_uncertain_files_get_final_recap(tmp_path: Path, capsys) -> None:
    _run(tmp_path, confidence=0.3)
    out = capsys.readouterr().out
    assert "UNCERTAIN" in out, "inline marker missing"
    assert "NOT PLACED (review these files manually):" in out, "final recap missing"
    assert "doc.txt" in out, "recap must name the file"
    assert "invoice 0.30" in out, "recap must show the top category with its probability"
    assert "other 0.10" in out, "recap must show the runner-up with its probability"


def test_confident_run_has_no_recap(tmp_path: Path, capsys) -> None:
    _run(tmp_path, confidence=0.95)
    out = capsys.readouterr().out
    assert "NOT PLACED" not in out


def test_bare_filename_passed_to_classifier(tmp_path: Path) -> None:
    """The classifier gets the bare filename, never the full path."""
    seen: list[str | None] = []

    class _Recording(_FixedClassifier):
        def classify(self, text, categories, *, name=None, descriptions=None):
            seen.append(name)
            return super().classify(text, categories, name=name)

    src_dir = tmp_path / "in"
    src_dir.mkdir()
    (src_dir / "bill.txt").write_text("Invoice #7 amount EUR 99")
    with patch("watfile.cli._build_classifier", return_value=_Recording(0.9)):
        main([str(src_dir), "-c", "invoice,other", "-o", str(tmp_path / "out"), "--no-batch"])
    assert seen == ["bill.txt"], f"expected bare filename, got {seen}"


def test_batch_names_are_bare_filenames(tmp_path: Path) -> None:
    seen: list = []

    class _Recording(_FixedClassifier):
        def classify_batch(self, texts, categories, *, names=None, descriptions=None):
            seen.append(list(names))
            return super().classify_batch(texts, categories, names=names)

    src_dir = tmp_path / "in"
    src_dir.mkdir()
    (src_dir / "a.txt").write_text("Invoice A")
    (src_dir / "b.txt").write_text("Invoice B")
    with patch("watfile.cli._build_classifier", return_value=_Recording(0.9)):
        main([str(src_dir), "-c", "invoice,other", "-o", str(tmp_path / "out"), "--batch", "5"])
    assert seen == [["a.txt", "b.txt"]], f"expected bare filenames, got {seen}"


def test_categories_and_directory_combined(tmp_path: Path) -> None:
    """-c + -d: target folder used as output root even if empty/missing."""
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    (src_dir / "doc.txt").write_text("Invoice #7 amount EUR 99")
    target = tmp_path / "fresh-target"  # does not exist yet
    with patch("watfile.cli._build_classifier", return_value=_FixedClassifier(0.9)):
        rc = main([str(src_dir), "-c", "invoice,other", "-d", str(target)])
    assert rc == 0
    assert (target / "invoice" / "doc.txt").is_symlink()


def test_neither_categories_nor_directory_errors(tmp_path: Path) -> None:
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    (src_dir / "doc.txt").write_text("content")
    with pytest.raises(SystemExit):
        main([str(src_dir), "-o", str(tmp_path / "out")])


def test_category_descriptions_reach_classifier(tmp_path: Path) -> None:
    """-c 'name:description,...' — descriptions land in the Choice criteria."""
    seen: dict = {}

    class _Recording(_FixedClassifier):
        def classify(self, text, categories, *, name=None, descriptions=None):
            seen["categories"] = list(categories)
            seen["descriptions"] = descriptions
            return super().classify(text, categories, name=name)

    src_dir = tmp_path / "in"
    src_dir.mkdir()
    (src_dir / "doc.txt").write_text("Invoice #7 amount EUR 99")
    with patch("watfile.cli._build_classifier", return_value=_Recording(0.9)):
        main([
            str(src_dir),
            "-c", "invoice:bills and payment requests,other",
            "-o", str(tmp_path / "out"),
            "--no-batch",
        ])
    assert seen["categories"] == ["invoice", "other"]
    assert seen["descriptions"] == {"invoice": "bills and payment requests"}


def test_parse_categories_descriptions() -> None:
    from watfile.cli import _parse_categories_arg

    names, descriptions = _parse_categories_arg("invoice:bills,donation,apartment:rental contracts")
    assert names == ["invoice", "donation", "apartment"]
    assert descriptions == {"invoice": "bills", "apartment": "rental contracts"}


def test_parse_quoted_descriptions_with_commas() -> None:
    from watfile.cli import _parse_categories_arg

    raw = 'invoice:"bills, payments, and refund requests",other:"misc, everything else",third'
    names, descriptions = _parse_categories_arg(raw)
    assert names == ["invoice", "other", "third"]
    assert descriptions == {
        "invoice": "bills, payments, and refund requests",
        "other": "misc, everything else",
    }
