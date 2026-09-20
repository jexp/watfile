"""Config resolution: env > ./.env > config.toml."""

import os
from pathlib import Path

import pytest

from watfile.config import Config, apply_config, load_config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No real env, no real .env, no real home config leak into these tests."""
    for var in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "WATFILE_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WATFILE_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.chdir(tmp_path)
    yield
    # apply_config() writes os.environ directly (beyond monkeypatch's tracking)
    for var in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL"):
        os.environ.pop(var, None)


def test_empty_when_nothing_configured() -> None:
    assert load_config() == Config()


def test_reads_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from_dotenv\n# comment\n")
    assert load_config().api_key == "from_dotenv"


def test_reads_toml_config(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('api_key = "from_toml"\nmodel = "jev-test"\n')
    config = load_config()
    assert config.api_key == "from_toml"
    assert config.model == "jev-test"


def test_env_beats_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from_dotenv\n")
    monkeypatch.setenv("TYPESAFE_API_KEY", "from_env")
    assert load_config().api_key == "from_env"


def test_dotenv_beats_toml(tmp_path: Path) -> None:
    # precedence: env > ./.env > config.toml (project-local overrides user-global)
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from_dotenv\n")
    (tmp_path / "config.toml").write_text('api_key = "from_toml"\n')
    assert load_config().api_key == "from_dotenv"


def test_apply_config_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_config(Config(api_key="k", model="m"))
    assert os.environ["TYPESAFE_API_KEY"] == "k"
    assert os.environ["TYPESAFE_DEFAULT_MODEL"] == "m"
