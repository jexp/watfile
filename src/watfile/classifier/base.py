"""Classifier abstraction — backend-agnostic file categorisation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Verdict:
    """Classification result for one document."""

    category: str
    confidence: float  # backend-reported certainty, 0..1
    probabilities: dict[str, float]  # category -> probability


class Classifier(ABC):
    """Backend contract. Implementations must be stateless per call so batching
    (classify_batch) can be added without changing the interface."""

    @abstractmethod
    def classify(self, text: str, categories: Sequence[str], *, name: str | None = None) -> Verdict:
        """Classify one document's text into exactly one of *categories*.

        *name* is the bare filename (no directory), passed as a weak hint
        alongside the content — never the full path.
        """
        ...

    def classify_batch(
        self, texts: Sequence[str], categories: Sequence[str], *, names: Sequence[str | None] | None = None
    ) -> list[Verdict]:
        """Classify many documents. Backends that pack multiple documents into
        one API call override this; the default loops classify().

        Must return one Verdict per input, in order.
        """
        name_list = list(names) if names is not None else [None] * len(texts)
        return [self.classify(t, categories, name=n) for t, n in zip(texts, name_list)]
