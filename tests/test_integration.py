"""Live integration tests against the TypeSafe API.

Run with TYPESAFE_API_KEY set (fetch from integration.env if present).
"""

import os
from pathlib import Path

import pytest

from watfile.classifier.jev import JevClassifier
from watfile.cli import main

pytestmark = pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY not set (check integration.env)",
)


def test_jev_classifies_invoice_text() -> None:
    verdict = JevClassifier().classify(
        "Invoice #2026-0815\nFrom: ACME GmbH\nTotal: EUR 1,200.00\nDue: 2026-10-01\nIBAN: DE89...",
        ["invoice", "donation", "apartment"],
    )
    assert verdict.category == "invoice"
    assert 0.0 <= verdict.confidence <= 1.0
    assert set(verdict.probabilities) == {"invoice", "donation", "apartment"}


def test_cli_end_to_end(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    incoming.mkdir()
    (incoming / "bill.txt").write_text("Invoice #7, amount EUR 99, due next week, pay to IBAN DE12...")
    (incoming / "charity.txt").write_text("Donation receipt: thank you for donating EUR 25 to charity Y.")

    out = tmp_path / "sorted"
    rc = main([str(incoming), "-c", "invoice,donation,apartment", "-o", str(out)])
    assert rc == 0
    assert (out / "invoice" / "bill.txt").exists()
    assert (out / "donation" / "charity.txt").exists()
