"""Small, dependency-free persistent cache used by the data providers.

The cache intentionally stores one atomic pickle envelope per key.  Pickle keeps
``pandas`` objects lossless (including NaNs and timestamp indexes), which is
important for financial data.  Consequently a cache directory must be treated
as trusted local application state and must never be populated with untrusted
files.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


@dataclass(frozen=True)
class CacheLookup(Generic[T]):
    """Result of looking up a cache key.

    ``found`` means the value may be used.  With ``allow_stale=True`` an
    expired value is returned with both ``found`` and ``stale`` set to true.
    """

    key: str
    found: bool
    value: T | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    stale: bool = False


@dataclass(frozen=True)
class _Envelope(Generic[T]):
    version: int
    key: str
    created_at: datetime
    expires_at: datetime | None
    value: T


class DiskCache:
    """Lossless disk cache with TTL, force-refresh and stale fallback support.

    Writes use ``os.replace`` so an interrupted process leaves either the old
    complete entry or the new complete entry.  Per-symbol/provider keys make a
    long market-data run naturally resumable.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        default_ttl: timedelta | float | int | None = timedelta(hours=24),
        namespace: str = "data",
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        self.root = Path(root).expanduser().resolve() / self._safe_component(namespace)
        self.default_ttl = self._coerce_ttl(default_ttl)
        self._now_fn = now_fn

    @staticmethod
    def _safe_component(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
        return cleaned or "cache"

    @staticmethod
    def _coerce_ttl(value: timedelta | float | int | None) -> timedelta | None:
        if value is None:
            return None
        ttl = value if isinstance(value, timedelta) else timedelta(seconds=float(value))
        if ttl.total_seconds() < 0:
            raise ValueError("cache TTL cannot be negative")
        return ttl

    def _now(self) -> datetime:
        value = self._now_fn()
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        hint = self._safe_component(key)[:48]
        return self.root / f"{hint}-{digest}.cache"

    def lookup(
        self,
        key: str,
        *,
        ttl: timedelta | float | int | None = None,
        force_refresh: bool = False,
        allow_stale: bool = False,
    ) -> CacheLookup[T]:
        """Look up ``key`` without confusing a cached ``None`` with a miss."""

        path = self._path_for(key)
        try:
            with path.open("rb") as handle:
                envelope = pickle.load(handle)  # noqa: S301 - trusted local cache only
        except Exception:
            # Cache entries are disposable local state. Corrupt bytes and
            # pickles referring to classes/modules from an older installation
            # should both degrade to a miss instead of aborting a screen.
            return CacheLookup(key=key, found=False)

        if not isinstance(envelope, _Envelope):
            return CacheLookup(key=key, found=False)
        if envelope.version != self.FORMAT_VERSION or envelope.key != key:
            return CacheLookup(key=key, found=False)

        effective_ttl = self._coerce_ttl(ttl) if ttl is not None else None
        expires_at = (
            envelope.created_at + effective_ttl
            if effective_ttl is not None
            else envelope.expires_at
        )
        expired = expires_at is not None and self._now() >= expires_at
        stale = force_refresh or expired
        usable = not stale or allow_stale
        return CacheLookup(
            key=key,
            found=usable,
            value=envelope.value if usable else None,
            created_at=envelope.created_at,
            expires_at=expires_at,
            stale=stale,
        )

    def get(
        self,
        key: str,
        default: T | None = None,
        **lookup_options: object,
    ) -> T | None:
        """Convenience wrapper returning ``default`` on a miss."""

        result = self.lookup(key, **lookup_options)
        return result.value if result.found else default

    def set(
        self,
        key: str,
        value: T,
        *,
        ttl: timedelta | float | int | None = None,
    ) -> T:
        """Atomically persist a value and return it."""

        selected_ttl = self.default_ttl if ttl is None else self._coerce_ttl(ttl)
        created_at = self._now()
        envelope = _Envelope(
            version=self.FORMAT_VERSION,
            key=key,
            created_at=created_at,
            expires_at=created_at + selected_ttl if selected_ttl is not None else None,
            value=value,
        )
        payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)

        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path_for(key)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", dir=self.root, delete=False
            ) as handle:
                temp_path = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        finally:
            if temp_path is not None:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass
        return value

    def get_or_compute(
        self,
        key: str,
        factory: Callable[[], T],
        *,
        ttl: timedelta | float | int | None = None,
        force_refresh: bool = False,
        stale_if_error: bool = True,
    ) -> T:
        """Return a cached value or compute it, optionally falling back stale."""

        hit = self.lookup(key, ttl=ttl, force_refresh=force_refresh)
        if hit.found:
            return hit.value  # type: ignore[return-value]

        try:
            return self.set(key, factory(), ttl=ttl)
        except Exception:
            if stale_if_error:
                stale = self.lookup(key, ttl=ttl, allow_stale=True)
                if stale.found:
                    return stale.value  # type: ignore[return-value]
            raise

    def invalidate(self, key: str) -> bool:
        """Remove exactly one entry; return whether it existed."""

        path = self._path_for(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True
