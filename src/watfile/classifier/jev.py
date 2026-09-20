"""TypeSafe AI Jev (System One) classifier backend."""

from typing import Sequence

from typesafe_sdk import Choice, TypeSafeClient

from .base import Classifier, Verdict

_QUESTION_KEY = "category"

#: All documents go into ONE state; each gets its own Choice question. The
#: question key encodes the document index so answers map back in order.
_BATCH_KEY_PREFIX = "doc"


class JevClassifier(Classifier):
    def __init__(self, model: str = "jev-latest") -> None:
        self._client = TypeSafeClient(model=model)

    def classify(self, text: str, categories: Sequence[str]) -> Verdict:
        question = self._question(categories)
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

    def classify_batch(self, texts: Sequence[str], categories: Sequence[str]) -> list[Verdict]:
        """Classify all *texts* in ONE system_one call.

        Documents are concatenated into one state with explicit
        `=== DOCUMENT N START/END ===` markers, and each question's
        instructions name its document number — without that binding the
        model cannot tell which document a question refers to and returns
        confidently wrong answers (measured: 1/4 vs 4/4 on fixtures).
        """
        if not texts:
            return []
        state = "\n\n".join(
            f"=== DOCUMENT {i} START ===\n{text}\n=== DOCUMENT {i} END ==="
            for i, text in enumerate(texts)
        )
        questions = {
            f"{_BATCH_KEY_PREFIX}{i}": Choice(
                instructions=(
                    f"Which folder/category does DOCUMENT {i} belong to? "
                    "Judge by its content, not other documents. Pick exactly one."
                ),
                criteria={cat: None for cat in categories},
            )
            for i in range(len(texts))
        }
        response = self._client.system_one(state=state, questions=questions)
        verdicts: list[Verdict] = []
        for i in range(len(texts)):
            answer = response.choices[f"{_BATCH_KEY_PREFIX}{i}"]
            verdicts.append(
                Verdict(
                    category=answer.choice,
                    confidence=answer.confidence,
                    probabilities=dict(answer.probabilities),
                )
            )
        return verdicts

    @staticmethod
    def _question(categories: Sequence[str]) -> Choice:
        return Choice(
            instructions=(
                "Which folder/category does this document belong to? "
                "Judge by content, not filename. Pick exactly one."
            ),
            criteria={cat: None for cat in categories},
        )
