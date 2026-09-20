from watfile.classifier.base import Classifier, Verdict
from watfile.classifier.multi import (
    MultiChunkClassifier,
    chunk_text_tokens,
    count_tokens,
)


class _FakeClassifier(Classifier):
    """Classifies by which keyword the chunk contains; probabilities from counts."""

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, text: str, categories) -> Verdict:
        self.calls += 1
        probs = {c: 0.1 for c in categories}
        for c in categories:
            probs[c] += 10 * text.lower().count(c.lower())
        total = sum(probs.values())
        probs = {c: p / total for c, p in probs.items()}
        winner = max(probs, key=probs.get)
        return Verdict(category=winner, confidence=probs[winner], probabilities=probs)


def test_chunk_text_tokens_respects_budget() -> None:
    text = " ".join(f"word{i}" for i in range(1000))
    parts = chunk_text_tokens(text, max_tokens_per_chunk=100)
    assert len(parts) > 1
    assert all(count_tokens(p) <= 100 for p in parts)
    # all words preserved in order
    assert " ".join(" ".join(p.split()) for p in parts).split() == text.split()


def test_chunk_text_tokens_short_text_single_chunk() -> None:
    assert chunk_text_tokens("short text", 100) == ["short text"]
    assert chunk_text_tokens("anything", 1000) == ["anything"]


def test_multichunk_fixed_count_aggregates() -> None:
    # many small chunks: majority biology -> aggregated winner biology
    text = "\n\n".join(["biology sample text"] * 6 + ["engineering sample text"] * 4)
    verdict = MultiChunkClassifier(_FakeClassifier(), chunks=5, tokens_per_chunk=100).classify(
        text, ["biology", "engineering", "art"]
    )
    assert verdict.category == "biology"
    assert verdict.probabilities["biology"] > verdict.probabilities["engineering"]


def test_multichunk_adaptive_stops_early_when_decisive() -> None:
    inner = _FakeClassifier()
    # chunk 1 (biology only) is decisive; the engineering tail must not be classified
    text = "\n\n".join(["biology " * 60] + ["engineering " * 60] * 5)
    clf = MultiChunkClassifier(inner, chunks=0, tokens_per_chunk=100, threshold=0.5)
    verdict = clf.classify(text, ["biology", "engineering"])
    assert inner.calls == 1, "should stop after first decisive chunk"
    assert verdict.category == "biology"


def test_multichunk_adaptive_extends_when_ambiguous() -> None:
    inner = _FakeClassifier()
    # every chunk is 50/50 mixed -> never decisive -> must keep extending
    # (~650 tokens total, 100-token chunks -> ~7 chunks)
    text = "\n\n".join(["biology engineering topic discussion analysis " * 4] * 12)
    parts = chunk_text_tokens(text, 100)
    assert len(parts) > 2, "test text must actually split"
    clf = MultiChunkClassifier(inner, chunks=0, tokens_per_chunk=100, threshold=0.99)
    clf.classify(text, ["biology", "engineering"])
    assert inner.calls > 1, "ambiguous decision should extend with more chunks"


def test_multichunk_adaptive_respects_max_chunks() -> None:
    inner = _FakeClassifier()
    text = "\n\n".join(["biology engineering topic discussion analysis " * 4] * 20)
    clf = MultiChunkClassifier(inner, chunks=0, tokens_per_chunk=50, threshold=1.1, max_chunks=3)
    clf.classify(text, ["biology", "engineering"])
    assert inner.calls == 3, "must stop at max_chunks even if never decisive"
