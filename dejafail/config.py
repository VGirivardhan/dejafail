"""Environment configuration. Secrets are read from .env and never printed."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv

DEFAULT_HINDSIGHT_URL = "https://api.hindsight.vectorize.io"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
REQUIRED_KEYS = ("HINDSIGHT_API_KEY", "GROQ_API_KEY")


class ConfigError(RuntimeError):
    """Raised when required settings are missing."""


@dataclass(frozen=True)
class Config:
    hindsight_api_key: str
    hindsight_url: str
    groq_api_key: str
    groq_model: str
    repo: str

    @property
    def bank_id(self) -> str:
        return f"dejafail-{self.repo}"

    def __repr__(self) -> str:  # keep keys out of logs and tracebacks
        return (
            f"Config(hindsight_url={self.hindsight_url!r}, "
            f"groq_model={self.groq_model!r}, repo={self.repo!r})"
        )


def load_config(env: Mapping[str, str] | None = None) -> Config:
    if env is None:
        load_dotenv()
        env = os.environ
    missing = [key for key in REQUIRED_KEYS if not env.get(key, "").strip()]
    if missing:
        raise ConfigError(
            f"Missing {', '.join(missing)} in .env (copy .env.example to .env and fill it in)"
        )
    return Config(
        hindsight_api_key=env["HINDSIGHT_API_KEY"].strip(),
        hindsight_url=env.get("HINDSIGHT_BASE_URL", "").strip() or DEFAULT_HINDSIGHT_URL,
        groq_api_key=env["GROQ_API_KEY"].strip(),
        groq_model=env.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL,
        repo=env.get("DEJAFAIL_REPO", "").strip() or "shopfront",
    )
