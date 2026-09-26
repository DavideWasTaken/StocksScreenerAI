from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from narrative_dislocation.data.cache import DiskCache


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


def test_disk_cache_distinguishes_cached_none_and_honours_ttl(tmp_path) -> None:
    clock = Clock()
    cache = DiskCache(tmp_path, default_ttl=60, now_fn=clock)

    cache.set("nullable", None)
    fresh = cache.lookup("nullable")
    assert fresh.found is True
    assert fresh.value is None
    assert fresh.stale is False

    clock.value += timedelta(seconds=61)
    expired = cache.lookup("nullable")
    assert expired.found is False
    assert expired.stale is True

    stale = cache.lookup("nullable", allow_stale=True)
    assert stale.found is True
    assert stale.stale is True
    assert stale.value is None


def test_get_or_compute_force_refresh_and_stale_on_error(tmp_path) -> None:
    clock = Clock()
    cache = DiskCache(tmp_path, default_ttl=30, now_fn=clock)
    calls: list[int] = []

    def build() -> dict[str, int]:
        calls.append(1)
        return {"version": len(calls)}

    assert cache.get_or_compute("key", build) == {"version": 1}
    assert cache.get_or_compute("key", build) == {"version": 1}
    assert cache.get_or_compute("key", build, force_refresh=True) == {"version": 2}

    clock.value += timedelta(seconds=31)

    def fail() -> dict[str, int]:
        raise RuntimeError("offline")

    assert cache.get_or_compute("key", fail, stale_if_error=True) == {"version": 2}
    with pytest.raises(RuntimeError, match="offline"):
        cache.get_or_compute("key", fail, stale_if_error=False)


def test_cache_invalidate_only_removes_requested_key(tmp_path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("one", 1)
    cache.set("two", 2)

    assert cache.invalidate("one") is True
    assert cache.invalidate("one") is False
    assert cache.lookup("one").found is False
    assert cache.get("two") == 2


def test_incompatible_pickle_degrades_to_cache_miss(tmp_path, monkeypatch) -> None:
    cache = DiskCache(tmp_path)
    cache.set("legacy", {"old": "value"})

    def incompatible_pickle(_handle):
        raise ModuleNotFoundError("old_provider_module")

    monkeypatch.setattr("narrative_dislocation.data.cache.pickle.load", incompatible_pickle)

    assert cache.lookup("legacy").found is False
