"""Confidence gating: files below --min-confidence are left in place."""

from pathlib import Path
from unittest.mock import patch

from watfile.classifier.base import Classifier, Verdict
from watfile.cli import main


class _FixedClassifier(Classifier):
    def __init__(self, confidence: float, category: str = "invoice") -> None:
        self._verdict = Verdict(
            category=category,
            confidence=confidence,
            probabilities={category: confidence},
        )

    def classify(self, text, categories) -> Verdict:
        return self._verdict

    def classify_batch(self, texts, categories):
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
