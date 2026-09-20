"""Laya backend tests. The live test needs the checkpoint (~1GB, downloaded
and cached on first use) and runs only when WATFILE_TEST_LAYA=1 is set."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from watfile.classifier.laya import _QUESTION_KEY, LayaClassifier
from watfile.extract import extract_text

FIXTURE_DIR = Path(__file__).parent / "fixture"


def _mock_agent(choice: str, probs: dict[str, float]) -> MagicMock:
    agent = MagicMock()
    agent.predict.return_value = {"answers": {_QUESTION_KEY: {"choice": choice, "probabilities": probs}}}
    return agent


def test_laya_maps_choice_answer_to_verdict() -> None:
    cats = ["invoice", "donation", "apartment"]
    probs = {"invoice": 0.7, "donation": 0.2, "apartment": 0.1}
    with patch("watfile.classifier.laya.laya") as laya_mod:
        laya_mod.load.return_value = _mock_agent("invoice", probs)
        verdict = LayaClassifier(model="test-checkpoint").classify("Invoice #42", cats)

    assert verdict.category == "invoice"
    assert verdict.confidence == 0.7
    assert verdict.probabilities == probs
    # question shape matches the upstream typed-decision schema
    _, questions = laya_mod.load.return_value.predict.call_args.args
    assert questions[_QUESTION_KEY]["type"] == "choice"
    assert questions[_QUESTION_KEY]["criteria"] == cats


def test_laya_truncates_state_to_context_limit() -> None:
    with patch("watfile.classifier.laya.laya") as laya_mod:
        laya_mod.load.return_value = _mock_agent("a", {"a": 1.0})
        LayaClassifier().classify("x" * 10_000, ["a", "b"])
    state, _ = laya_mod.load.return_value.predict.call_args.args
    assert len(state) <= 2_500


@pytest.mark.skipif(
    os.environ.get("WATFILE_TEST_LAYA") != "1",
    reason="set WATFILE_TEST_LAYA=1 to run local MLX inference (downloads ~1GB checkpoint once)",
)
def test_laya_classifies_computerscience_fixture() -> None:
    text = extract_text(FIXTURE_DIR / "2609.19155v1.pdf")
    verdict = LayaClassifier().classify(text, ["computerscience", "biology", "engineering", "astrophysics"])
    assert verdict.category in {"computerscience", "engineering", "astrophysics", "biology"}
    assert 0.0 <= verdict.confidence <= 1.0
