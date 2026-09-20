from unittest.mock import MagicMock, patch

from watfile.classifier.jev import _QUESTION_KEY, JevClassifier


def _mock_response(choice: str, confidence: float, probabilities: dict[str, float]) -> MagicMock:
    answer = MagicMock(choice=choice, confidence=confidence, probabilities=probabilities)
    response = MagicMock()
    response.choices = {_QUESTION_KEY: answer}
    return response


def test_jev_classifier_maps_choice_to_verdict() -> None:
    categories = ["invoice", "donation", "apartment"]
    response = _mock_response("invoice", 0.92, {"invoice": 0.92, "donation": 0.05, "apartment": 0.03})
    with patch("watfile.classifier.jev.TypeSafeClient") as client_cls:
        client_cls.return_value.system_one.return_value = response
        verdict = JevClassifier().classify("Invoice #42 due 2026-10-01", categories)

    assert verdict.category == "invoice"
    assert verdict.confidence == 0.92
    assert verdict.probabilities["invoice"] == 0.92
    # categories passed as Choice criteria
    question = client_cls.return_value.system_one.call_args.kwargs["questions"][_QUESTION_KEY]
    assert set(question.criteria) == set(categories)


def _batch_response(per_doc: list[tuple[str, float]]) -> MagicMock:
    """Fake batch response: choices keyed doc0..docN."""
    response = MagicMock()
    response.choices = {
        f"doc{i}": MagicMock(choice=cat, confidence=conf, probabilities={cat: conf})
        for i, (cat, conf) in enumerate(per_doc)
    }
    return response


def test_jev_batch_one_call_maps_each_answer() -> None:
    cats = ["invoice", "donation", "apartment"]
    per_doc = [("invoice", 0.99), ("donation", 0.95), ("apartment", 0.91)]
    with patch("watfile.classifier.jev.TypeSafeClient") as client_cls:
        client_cls.return_value.system_one.return_value = _batch_response(per_doc)
        verdicts = JevClassifier().classify_batch(["inv text", "don text", "apt text"], cats)

    assert client_cls.return_value.system_one.call_count == 1, "batch must be ONE call"
    assert [v.category for v in verdicts] == ["invoice", "donation", "apartment"]
    call = client_cls.return_value.system_one.call_args
    state = call.kwargs["state"]
    questions = call.kwargs["questions"]
    # explicit document markers required for the model to bind questions
    assert "=== DOCUMENT 0 START ===" in state and "=== DOCUMENT 2 END ===" in state
    assert set(questions) == {"doc0", "doc1", "doc2"}
    assert "DOCUMENT 1" in questions["doc1"].instructions


def test_jev_batch_empty_input() -> None:
    with patch("watfile.classifier.jev.TypeSafeClient"):
        assert JevClassifier().classify_batch([], ["a", "b"]) == []
