"""Laya backend tests, runtime-agnostic (mlx / coreml / torch).

Mocked tests stub the runtime's load(); they run on any machine because the
stub injects the agent directly. The live test needs a checkpoint (downloaded
and cached on first use) and runs only when WATFILE_TEST_LAYA=1 is set."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from watfile.classifier.laya import _QUESTION_KEY, LayaClassifier
from watfile.extract import extract_text

FIXTURE_DIR = Path(__file__).parent / "fixture"


def _mock_agent(choice: str, probs: dict[str, float]) -> MagicMock:
    agent = MagicMock()
    agent.predict.return_value = {"answers": {_QUESTION_KEY: {"choice": choice, "probabilities": probs}}}
    return agent


@pytest.fixture
def stubbed_agent(monkeypatch: pytest.MonkeyPatch):
    """Bypass runtime loading entirely: inject a mock agent lazily."""
    agent_holder = {}

    def fake_agent_or_load(self):
        if self._agent is None:
            self._agent = agent_holder.setdefault("agent", _mock_agent("invoice", {
                "invoice": 0.7, "donation": 0.2, "apartment": 0.1
            }))
        return self._agent

    monkeypatch.setattr(LayaClassifier, "_agent_or_load", fake_agent_or_load)
    return agent_holder


def test_laya_maps_choice_answer_to_verdict(stubbed_agent) -> None:
    cats = ["invoice", "donation", "apartment"]
    verdict = LayaClassifier(model="test-checkpoint", runtime="mlx").classify("Invoice #42", cats)
    assert verdict.category == "invoice"
    assert verdict.confidence == 0.7
    assert verdict.probabilities["invoice"] == 0.7


def test_laya_question_shape_matches_upstream_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    cats = ["invoice", "donation", "apartment"]
    agent = _mock_agent("invoice", {"invoice": 0.7, "donation": 0.2, "apartment": 0.1})
    monkeypatch.setattr(LayaClassifier, "_agent_or_load", lambda self: agent)
    LayaClassifier(model="test-checkpoint", runtime="torch").classify("Invoice #42", cats)
    _, questions = agent.predict.call_args.args
    assert questions[_QUESTION_KEY]["type"] == "choice"
    assert questions[_QUESTION_KEY]["criteria"] == cats


def test_laya_truncates_state_to_context_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _mock_agent("a", {"a": 1.0})
    monkeypatch.setattr(LayaClassifier, "_agent_or_load", lambda self: agent)
    LayaClassifier(runtime="coreml").classify("x" * 10_000, ["a", "b"])
    state, _ = agent.predict.call_args.args
    assert len(state) <= 3_400


def test_laya_runtime_override_respected() -> None:
    assert LayaClassifier(runtime="torch").runtime == "torch"
    assert LayaClassifier(runtime="coreml").runtime == "coreml"


@pytest.mark.skipif(
    os.environ.get("WATFILE_TEST_LAYA") != "1",
    reason="set WATFILE_TEST_LAYA=1 to run local inference (downloads checkpoint once)",
)
def test_laya_classifies_computerscience_fixture() -> None:
    text = extract_text(FIXTURE_DIR / "2609.19155v1.pdf")
    verdict = LayaClassifier().classify(text, ["computerscience", "biology", "engineering", "astrophysics"])
    assert verdict.category in {"computerscience", "engineering", "astrophysics", "biology"}
    assert 0.0 <= verdict.confidence <= 1.0
