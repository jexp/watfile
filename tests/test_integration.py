"""Live integration tests against the TypeSafe API.

Run with TYPESAFE_API_KEY set (fetch from integration.env if present).
"""

from pathlib import Path

import pytest

from watfile.classifier.jev import JevClassifier
from watfile.cli import main
from watfile.config import api_key_available, apply_config, load_config

pytestmark = pytest.mark.skipif(
    not api_key_available(),
    reason="no TYPESAFE_API_KEY in env, ./.env, or ~/.config/watfile/config.toml",
)


def _ensure_key_in_env() -> None:
    apply_config(load_config())


def test_jev_classifies_invoice_text() -> None:
    _ensure_key_in_env()
    verdict = JevClassifier().classify(
        "Invoice #2026-0815\nFrom: ACME GmbH\nTotal: EUR 1,200.00\nDue: 2026-10-01\nIBAN: DE89...",
        ["invoice", "donation", "apartment"],
    )
    assert verdict.category == "invoice"
    assert 0.0 <= verdict.confidence <= 1.0
    assert set(verdict.probabilities) == {"invoice", "donation", "apartment"}


def test_cli_end_to_end(tmp_path: Path) -> None:
    _ensure_key_in_env()
    incoming = tmp_path / "in"
    incoming.mkdir()
    (incoming / "bill.txt").write_text("Invoice #7, amount EUR 99, due next week, pay to IBAN DE12...")
    (incoming / "charity.txt").write_text("Donation receipt: thank you for donating EUR 25 to charity Y.")

    out = tmp_path / "sorted"
    rc = main([str(incoming), "-c", "invoice,donation,apartment", "-o", str(out)])
    assert rc == 0
    # default placement is symlink: link exists in category folder, original stays
    assert (out / "invoice" / "bill.txt").is_symlink()
    assert (out / "donation" / "charity.txt").is_symlink()
    assert (incoming / "bill.txt").exists()
    assert (incoming / "charity.txt").exists()


def test_cli_move_mode(tmp_path: Path) -> None:
    _ensure_key_in_env()
    incoming = tmp_path / "in"
    incoming.mkdir()
    (incoming / "bill.txt").write_text("Invoice #7, amount EUR 99, due next week, pay to IBAN DE12...")
    out = tmp_path / "sorted"
    rc = main([str(incoming), "-c", "invoice,donation,apartment", "-o", str(out), "-m"])
    assert rc == 0
    dest = out / "invoice" / "bill.txt"
    assert dest.exists() and not dest.is_symlink()
    assert not (incoming / "bill.txt").exists()


def test_cli_batch_mode(tmp_path: Path) -> None:
    """--batch N classifies N files per API call."""
    _ensure_key_in_env()
    incoming = tmp_path / "in"
    incoming.mkdir()
    (incoming / "bill.txt").write_text("Invoice #7, amount EUR 99, due next week, pay to IBAN DE12...")
    (incoming / "charity.txt").write_text("Donation receipt: thank you for donating EUR 25 to charity Y.")
    (incoming / "lease.txt").write_text("Mietvertrag: 3-Zimmer-Wohnung, Grundmiete 1.180 EUR, Kaution 2.800 EUR.")
    out = tmp_path / "sorted"
    rc = main([str(incoming), "-c", "invoice,donation,apartment", "-o", str(out), "--batch", "3"])
    assert rc == 0
    assert (out / "invoice" / "bill.txt").is_symlink()
    assert (out / "donation" / "charity.txt").is_symlink()
    assert (out / "apartment" / "lease.txt").is_symlink()
