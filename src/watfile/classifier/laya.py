"""Local MLX classifier backend using laya-mlx (typed decision models).

Laya runs natively on Apple Silicon via MLX — ~13ms per decision, fully local,
no server needed. Question/answer shape mirrors the TypeSafe Jev API
(`choice` with criteria → probabilities over categories).

    uv pip install laya-mlx   (already a watfile dependency)
    # first classify() downloads the checkpoint (~1GB), then it's cached

Checkpoint and limits (https://huggingface.co/aac6fef):
    aac6fef/laya-mlx                 English, 421M, 512-token context
    aac6fef/laya-multilingual-mlx    multilingual, 322M, 1024-token context
    aac6fef/laya-typed-decisions-mlx English, 421M, 1024-token context
"""

from __future__ import annotations

import os
from typing import Sequence

import laya_mlx as laya

from .base import Classifier, Verdict

#: Default checkpoint: typed-decisions has the larger 1024-token context.
DEFAULT_MODEL = "aac6fef/laya-typed-decisions-mlx"

#: laya's encoder context is 512-1024 tokens. When used bare (no MultiChunk
#: wrapper), clamp the state by estimated tokens so the encoder isn't overflowed.
#: Chars are a rough proxy (~4 chars/token); MultiChunkClassifier sizes by
#: real tiktoken counts instead.
_MAX_STATE_CHARS = 3_400

_QUESTION_KEY = "category"


class LayaClassifier(Classifier):
    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or os.environ.get("LAYA_MODEL") or DEFAULT_MODEL
        self._agent = None  # lazy: load downloads the checkpoint on first use

    def _agent_or_load(self):
        if self._agent is None:
            self._agent = laya.load(self._model_name, dtype="float16")
        return self._agent

    def classify(self, text: str, categories: Sequence[str]) -> Verdict:
        state = text[:_MAX_STATE_CHARS]
        questions = {
            _QUESTION_KEY: {
                "type": "choice",
                "instructions": (
                    "Which category does this document belong to? "
                    "Judge by content, not filename."
                ),
                "criteria": list(categories),
            }
        }
        result = self._agent_or_load().predict(state, questions)
        answer = result["answers"][_QUESTION_KEY]
        # answer: {"choice": <label>, "probabilities": {...}, ...} (upstream schema)
        probabilities = {c: float(p) for c, p in answer["probabilities"].items()}
        category = answer["choice"]
        return Verdict(
            category=category,
            confidence=probabilities.get(category, 0.0),
            probabilities=probabilities,
        )
