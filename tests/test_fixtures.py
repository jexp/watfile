"""Tests against real arxiv PDFs in tests/fixture.

Ground truth comes from the arXiv subject tag in each paper's header:
    2609.19155v1  cs.CL      -> computerscience
    2609.19235v1  astro-ph   -> astrophysics
    2609.19320v1  eess.SY    -> engineering
    2609.19371v1  q-bio.TO   -> biology
"""

import os
import re
from pathlib import Path

import pytest

from watfile.classifier.jev import JevClassifier
from watfile.extract import extract_text

FIXTURE_DIR = Path(__file__).parent / "fixture"

CATEGORIES = ["computerscience", "biology", "engineering", "astrophysics"]

EXPECTED = {
    "2609.19155v1.pdf": "computerscience",
    "2609.19235v1.pdf": "astrophysics",
    "2609.19320v1.pdf": "engineering",
    "2609.19371v1.pdf": "biology",
}

_SUBJECT_RE = re.compile(r"\[([a-z-]+(?:\.[A-Z]+)?)\]", re.IGNORECASE)


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_fixture_extraction_contains_subject_tag(filename: str) -> None:
    path = FIXTURE_DIR / filename
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")
    text = extract_text(path)
    assert len(text) > 500, "extraction returned suspiciously little text"
    # arXiv subject tag appears in the header of page 1
    assert _SUBJECT_RE.search(text[:2000]), "no [subject] tag found in header"


@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY not set",
)
@pytest.mark.parametrize("filename,expected", sorted(EXPECTED.items()))
def test_jev_classifies_fixture_correctly(filename: str, expected: str) -> None:
    text = extract_text(FIXTURE_DIR / filename)
    verdict = JevClassifier().classify(text, CATEGORIES)
    assert verdict.category == expected, (
        f"{filename}: got {verdict.category} (conf {verdict.confidence:.2f}), "
        f"probabilities={verdict.probabilities}"
    )
