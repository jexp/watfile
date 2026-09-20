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
    (classify_many) can be added without changing the interface."""

    @abstractmethod
    def classify(self, text: str, categories: Sequence[str]) -> Verdict:
        """Classify one document's text into exactly one of *categories*."""
        ...
