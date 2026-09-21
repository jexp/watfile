"""Multi-chunk classification: classify a document in overlapping-free text
chunks and aggregate the per-chunk probabilities.

Useful for backends with small context windows (laya: 512-1024 tokens): instead
of one decision on a truncated head, the backend sees the whole extract split
into chunks and the mean of per-chunk probability vectors decides. Noisy
chunks (headers, references) get outvoted by content-bearing chunks.

Wraps any Classifier, so it composes: MultiChunkClassifier(JevClassifier(), 5).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import tiktoken

from .base import Classifier, Verdict

#: proxy tokenizer for budget math; laya checkpoints use their own HF
#: tokenizers, but counts track closely enough for chunk sizing.
@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def chunk_text_tokens(text: str, max_tokens_per_chunk: int, overlap: int = 0) -> list[str]:
    """Split *text* into chunks of at most *max_tokens_per_chunk* tokens.

    Chunks break at sentence/whitespace boundaries where possible; token
    counts are computed with tiktoken (o200k_base proxy). *overlap* tokens
    of context carry across chunk borders.
    """
    if max_tokens_per_chunk <= 0:
        raise ValueError("max_tokens_per_chunk must be positive")
    text = text.strip()
    if not text:
        return [text] if text else []
    enc = _encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens_per_chunk:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens_per_chunk, len(tokens))
        # snap the end back to whitespace so we don't cut mid-word
        chunk = enc.decode(tokens[start:end])
        if end < len(tokens):
            cut = chunk.rfind(" ")
            newline = chunk.rfind("\n")
            boundary = max(cut, newline)
            if boundary > max_tokens_per_chunk // 4:  # don't waste >3/4 of budget
                chunk = chunk[:boundary].rstrip()
                end = start + len(enc.encode(chunk))
        chunks.append(chunk.strip())
        if end >= len(tokens):
            break
        start = end - overlap if overlap > 0 else end
    return [c for c in chunks if c]


def chunk_text(text: str, n_chunks: int) -> list[str]:
    """Split *text* into *n_chunks* parts, breaking at whitespace boundaries.

    Returns fewer/larger chunks when the text is too short to split.
    """
    if n_chunks <= 1:
        return [text]
    text = text.strip()
    if not text:
        return [text]
    # not enough words to split into n_chunks meaningful parts — one chunk
    if len(text.split()) < n_chunks * 2:
        return [text]
    target = len(text) / n_chunks
    chunks: list[str] = []
    start = 0
    for i in range(1, n_chunks):
        # ideal split point, then snap to the nearest whitespace
        ideal = round(i * target)
        window_lo, window_hi = max(start + 1, ideal - 400), min(len(text), ideal + 400)
        window = text[window_lo:window_hi]
        if not window.strip():
            break_pos = ideal
        else:
            # nearest whitespace to the ideal point within the window
            offsets = [j for j, c in enumerate(window) if c.isspace()]
            if not offsets:
                break_pos = ideal
            else:
                rel = min(offsets, key=lambda j: abs(window_lo + j - ideal))
                break_pos = window_lo + rel
        if break_pos <= start:
            break_pos = start + 1
        chunks.append(text[start:break_pos].strip())
        start = break_pos
    chunks.append(text[start:].strip())
    return [c for c in chunks if c]


class MultiChunkClassifier(Classifier):
    """Classify via *inner* on text chunks, aggregate by mean probability.

    With chunks=0 (adaptive mode): classify chunk 1; if the winner's
    aggregated probability is below *threshold*, extend by one chunk at a
    time (up to *max_chunks*) and re-aggregate — decisive documents stop
    after 1 call, ambiguous ones gather more evidence.

    confidence is the aggregated probability of the winning category
    (more honest than the mean of per-chunk confidences, which would
    average away disagreement between chunks).
    """

    #: default decisiveness threshold for adaptive mode
    DEFAULT_THRESHOLD = 0.5

    #: default token budget per chunk. 200 tokens: small enough that a 2-page
    #: extract yields ~8-10 evidence units for majority voting, large enough
    #: that each chunk carries a coherent topic sentence or two. Measured on
    #: the arXiv fixtures: 850-token chunks give 2/4, 200-token give 3/4.
    DEFAULT_TOKENS_PER_CHUNK = 200

    def __init__(
        self,
        inner: Classifier,
        chunks: int = 5,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_chunks: int = 10,
        tokens_per_chunk: int = DEFAULT_TOKENS_PER_CHUNK,
    ) -> None:
        if chunks < 0:
            raise ValueError("chunks must be >= 0 (0 = adaptive)")
        self._inner = inner
        self._chunks = chunks
        self._threshold = threshold
        self._max_chunks = max_chunks
        self._tokens_per_chunk = tokens_per_chunk

    def classify(
        self,
        text: str,
        categories: Sequence[str],
        *,
        name: str | None = None,
        descriptions: dict[str, str] | None = None,
    ) -> Verdict:
        parts = chunk_text_tokens(text, self._tokens_per_chunk)

        if self._chunks > 0:
            # fixed mode: classify exactly N chunks
            verdicts = [
                self._inner.classify(p, categories, name=name, descriptions=descriptions)
                for p in parts[: self._chunks]
            ]
            return self._aggregate(verdicts, categories)

        # adaptive: classify chunk-by-chunk, extend only while the aggregated
        # decision is not decisive. Each chunk is classified at most once.
        n_max = min(len(parts), self._max_chunks)
        if n_max == 0:
            return self._aggregate([], categories)
        sums = {c: 0.0 for c in categories}
        aggregated: Verdict | None = None
        for n in range(1, n_max + 1):
            verdict = self._inner.classify(parts[n - 1], categories, name=name, descriptions=descriptions)
            for cat, p in verdict.probabilities.items():
                if cat in sums:
                    sums[cat] += p
            probabilities = {cat: s / n for cat, s in sums.items()}
            category = max(probabilities, key=probabilities.get)
            aggregated = Verdict(
                category=category, confidence=probabilities[category], probabilities=probabilities
            )
            if aggregated.confidence >= self._threshold:
                return aggregated
        assert aggregated is not None
        return aggregated

    def _aggregate(self, verdicts: Sequence[Verdict], categories: Sequence[str]) -> Verdict:
        sums: dict[str, float] = {c: 0.0 for c in categories}
        for verdict in verdicts:
            for cat, p in verdict.probabilities.items():
                if cat in sums:
                    sums[cat] += p
        n = max(1, len(verdicts))
        probabilities = {cat: s / n for cat, s in sums.items()}
        category = max(probabilities, key=probabilities.get)
        return Verdict(
            category=category,
            confidence=probabilities[category],
            probabilities=probabilities,
        )
