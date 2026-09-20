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
