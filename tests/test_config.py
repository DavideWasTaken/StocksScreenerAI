from __future__ import annotations

from pathlib import Path

import pytest

from narrative_dislocation.config import AppConfig


def test_env_configuration_preserves_zero_as_real_value(monkeypatch, tmp_path):
    monkeypatch.setenv("ND_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ND_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("ND_AI_WEB_SEARCH", "false")
    monkeypatch.setenv("ND_MIN_DRAWDOWN", "0.30")

    config = AppConfig.from_env()

    assert config.cache_dir == tmp_path / "cache"
    assert config.output_dir == tmp_path / "out"
    assert config.ai_web_search is False
    assert config.min_drawdown == pytest.approx(0.30)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="min_drawdown"):
        AppConfig(min_drawdown=1.0).validate()


def test_prepare_directories_expands_user_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    config = AppConfig(cache_dir=Path("~/cache"), output_dir=Path("~/outputs"))

    config.prepare_directories()

    assert config.cache_dir == (tmp_path / "cache").resolve()
    assert config.output_dir == (tmp_path / "outputs").resolve()
    assert config.cache_dir.is_dir()
    assert config.output_dir.is_dir()
