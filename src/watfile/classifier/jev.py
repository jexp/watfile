"""TypeSafe AI Jev (System One) classifier backend."""

from typing import Sequence

from typesafe_sdk import Choice, TypeSafeClient

from .base import Classifier, Verdict

_QUESTION_KEY = "category"


class JevClassifier(Classifier):
    def __init__(self, model: str = "jev-latest") -> None:
        self._client = TypeSafeClient(model=model)

    def classify(self, text: str, categories: Sequence[str]) -> Verdict:
        question = Choice(
            instructions=(
                "Which folder/category does this document belong to? "
                "Judge by content, not filename. Pick exactly one."
            ),
            criteria={cat: None for cat in categories},
        )
        response = self._client.system_one(
            state={"document": text},
            questions={_QUESTION_KEY: question},
        )
        answer = response.choices[_QUESTION_KEY]
        return Verdict(
            category=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
        )
