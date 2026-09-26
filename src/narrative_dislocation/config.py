"""Runtime configuration with environment and CLI-friendly defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


DEFAULT_SECTOR_BENCHMARKS: dict[str, str] = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Utilities": "XLU",
}


@dataclass(slots=True)
class AppConfig:
    """All mutable policy choices for a screening run.

    Financial thresholds are deliberately centralized so the first version can
    be tuned after observing real results without rewriting provider code.
    """

    cache_dir: Path = Path(".cache/narrative_dislocation")
    output_dir: Path = Path("outputs")
    cache_ttl_hours: float = 24.0
    min_market_cap: float = 5_000_000_000.0
    min_drawdown: float = 0.25
    price_history_years: int = 6
    benchmark_ticker: str = "SPY"
    sector_benchmarks: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SECTOR_BENCHMARKS)
    )
    final_top: int = 20
    ai_candidates: int = 20
    report_top: int = 10
    openai_model: str = "gpt-5-mini"
    ai_web_search: bool = True

    @classmethod
    def from_env(cls) -> AppConfig:
        config = cls(
            cache_dir=Path(os.getenv("ND_CACHE_DIR", ".cache/narrative_dislocation")),
            output_dir=Path(os.getenv("ND_OUTPUT_DIR", "outputs")),
            cache_ttl_hours=_env_float("ND_CACHE_TTL_HOURS", 24.0),
            min_market_cap=_env_float("ND_MIN_MARKET_CAP", 5_000_000_000.0),
            min_drawdown=_env_float("ND_MIN_DRAWDOWN", 0.25),
            price_history_years=_env_int("ND_PRICE_HISTORY_YEARS", 6),
            benchmark_ticker=os.getenv("ND_BENCHMARK", "SPY"),
            final_top=_env_int("ND_FINAL_TOP", 20),
            ai_candidates=_env_int("ND_AI_CANDIDATES", 20),
            report_top=_env_int("ND_REPORT_TOP", 10),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            ai_web_search=_env_bool("ND_AI_WEB_SEARCH", True),
        )
        config.validate()
        return config

    def with_overrides(self, **values: Any) -> AppConfig:
        updated = replace(
            self, **{key: value for key, value in values.items() if value is not None}
        )
        updated.validate()
        return updated

    def validate(self) -> None:
        if self.min_market_cap <= 0:
            raise ValueError("min_market_cap must be positive")
        if not 0 < self.min_drawdown < 1:
            raise ValueError("min_drawdown must be between 0 and 1")
        if self.cache_ttl_hours < 0:
            raise ValueError("cache_ttl_hours cannot be negative")
        for name in ("final_top", "ai_candidates", "report_top"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def prepare_directories(self) -> None:
        self.cache_dir = self.cache_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_openai_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


def load_dotenv_if_present(path: Path | None = None) -> bool:
    """Load a local .env without making importing this package side-effectful."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    candidate = path or Path(".env")
    return bool(load_dotenv(candidate, override=False))
