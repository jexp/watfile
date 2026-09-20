"""Aggregation tests on realistic prose (see texts.py)."""

from texts import APARTMENT_LETTER, DONATION_RECEIPT, INVOICE, PAPER_BIO_HEAVY, PAPER_CS_HEAVY
from watfile.classifier.base import Classifier, Verdict
from watfile.classifier.multi import MultiChunkClassifier, chunk_text_tokens, count_tokens


class _KeywordClassifier(Classifier):
    """Weighted keyword scoring — a crude but deterministic stand-in that
    behaves like a real classifier: right on topic-dense text, weaker on
    topic-mixed chunks."""

    KEYWORDS = {
        "invoice": ["rechnung", "zwischensumme", "umsatzsteuer", "zahlung", "iban",
                    "position 1", "skonto", "bestellung", "netto"],
        "donation": ["spende", "hilfe", "verein", "wohltätig", "gemeinnützig",
                     "finanzamt", "bestätigung", "projekt"],
        "apartment": ["mietvertrag", "wohnung", "mieter", "vermieter", "miete",
                      "kaution", "kündigungsfrist", "betriebskosten", "qm"],
        "computerscience": ["attention", "model", "inference", "transformer",
                            "benchmark", "training", "algorithm", "pruning",
                            "perplexity", "checkpoints", "gradient", "entropy"],
        "biology": ["microbiome", "taxa", "patients", "immunity", "metagenome",
                    "bacterial", "cohort", "melanoma", "t-cell", "survival",
                    "disease"],
    }

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, text: str, categories) -> Verdict:
        self.calls += 1
        low = text.lower()
        scores = {c: 1.0 for c in categories}
        for cat in categories:
            for kw in self.KEYWORDS.get(cat, []):
                scores[cat] += 5 * low.count(kw)
        total = sum(scores.values())
        probs = {c: s / total for c, s in scores.items()}
        winner = max(probs, key=probs.get)
        return Verdict(category=winner, confidence=probs[winner], probabilities=probs)


def test_realistic_documents_classified_single_chunk() -> None:
    clf = _KeywordClassifier()
    cases = [
        (INVOICE, "invoice"),
        (DONATION_RECEIPT, "donation"),
        (APARTMENT_LETTER, "apartment"),
        (PAPER_CS_HEAVY, "computerscience"),
        (PAPER_BIO_HEAVY, "biology"),
    ]
    cats = ["invoice", "donation", "apartment", "computerscience", "biology"]
    for text, expected in cases:
        verdict = clf.classify(text, cats)
        assert verdict.category == expected, f"{expected}: got {verdict.category}"


def test_realistic_documents_multi_chunk_still_correct() -> None:
    cats = ["invoice", "donation", "apartment", "computerscience", "biology"]
    cases = [
        (INVOICE, "invoice"),
        (DONATION_RECEIPT, "donation"),
        (APARTMENT_LETTER, "apartment"),
        (PAPER_CS_HEAVY, "computerscience"),
        (PAPER_BIO_HEAVY, "biology"),
    ]
    for text, expected in cases:
        for chunks in (2, 3, 5):
            clf = MultiChunkClassifier(_KeywordClassifier(), chunks=chunks, tokens_per_chunk=200)
            verdict = clf.classify(text, cats)
            assert verdict.category == expected, f"{expected} @ chunks={chunks}: got {verdict.category}"


def test_adaptive_uses_fewer_calls_on_clear_documents() -> None:
    cats = ["invoice", "donation", "apartment"]
    # invoice is unambiguous throughout: first chunk should already be decisive
    decisive = _KeywordClassifier()
    MultiChunkClassifier(decisive, chunks=0, tokens_per_chunk=200, threshold=0.5).classify(
        INVOICE, cats
    )
    # ambiguous doc: every chunk mixed -> must use more chunks
    mixed = "\n\n".join([INVOICE[:400], APARTMENT_LETTER[:400]] * 3)
    ambiguous = _KeywordClassifier()
    MultiChunkClassifier(ambiguous, chunks=0, tokens_per_chunk=200, threshold=0.9).classify(
        mixed, cats
    )
    assert decisive.calls < ambiguous.calls


def test_token_chunks_of_realistic_text_are_substantial() -> None:
    for text in (INVOICE, PAPER_CS_HEAVY):
        parts = chunk_text_tokens(text, 150)
        assert len(parts) >= 2
        assert all(0 < count_tokens(p) <= 150 for p in parts)
        # content preserved: every original word appears across chunks
        original_words = set(text.split())
        chunked_words = set(" ".join(parts).split())
        assert original_words <= chunked_words
