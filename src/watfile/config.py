"""Configuration for watfile: API key and model resolution.

Resolution order (first wins):
  1. TYPESAFE_API_KEY environment variable
  2. .env file in the current directory (gitignored)
  3. ~/.config/watfile/config.toml  (XDG config home; $WATFILE_CONFIG overrides)
     keys: api_key, base_url, model

Uses stdlib tomllib only (Python 3.11+), no extra dependency.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: env var the TypeSafe SDK itself reads; used as source and final sink.
API_KEY_ENV = "TYPESAFE_API_KEY"

#: project-local env file, expected gitignored.
DOTENV_FILE = ".env"

#: config file name inside the config dir.
CONFIG_FILENAME = "config.toml"


def default_config_path() -> Path:
    """XDG-style config path: $XDG_CONFIG_HOME/watfile/config.toml (default ~/.config/...)."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "watfile" / CONFIG_FILENAME


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        print(f"warning: ignoring invalid config {path}: {exc}", file=sys.stderr)
        return {}


@dataclass(frozen=True)
class Config:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    laya_model: str | None = None


#: TOML config uses friendly field names (api_key/base_url/model);
#: .env and environment use the SDK's variable names.
_TOML_TO_ENV = {
    "api_key": API_KEY_ENV,
    "base_url": "TYPESAFE_BASE_URL",
    "model": "TYPESAFE_DEFAULT_MODEL",
    "laya_model": "LAYA_MODEL",
}


def load_config() -> Config:
    """Merge TOML config, ./.env, and environment into a Config (env wins)."""
    values: dict[str, str] = {}

    config_path = Path(os.environ.get("WATFILE_CONFIG") or default_config_path())
    for key, value in _load_toml(config_path).items():
        if isinstance(value, str) and key in _TOML_TO_ENV:
            values[_TOML_TO_ENV[key]] = value

    dotenv_path = Path(DOTENV_FILE)
    if dotenv_path.is_file():
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # .env overrides TOML (project-local beats user-global), but not real env
            if key.strip() not in os.environ:
                values[key.strip()] = value.strip().strip("'\"")

    # real environment always wins
    for key in (API_KEY_ENV, "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "LAYA_MODEL"):
        if os.environ.get(key):
            values[key] = os.environ[key]

    return Config(
        api_key=values.get(API_KEY_ENV),
        base_url=values.get("TYPESAFE_BASE_URL"),
        model=values.get("TYPESAFE_DEFAULT_MODEL"),
        laya_model=values.get("LAYA_MODEL"),
    )


def apply_config(config: Config) -> None:
    """Export resolved values into the environment so the typesafe-sdk picks them up."""
    if config.api_key:
        os.environ[API_KEY_ENV] = config.api_key
    if config.base_url:
        os.environ["TYPESAFE_BASE_URL"] = config.base_url
    if config.model:
        os.environ["TYPESAFE_DEFAULT_MODEL"] = config.model


def api_key_available() -> bool:
    """True if a key is resolvable from env, ./.env, or the config file."""
    return load_config().api_key is not None
