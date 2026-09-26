from __future__ import annotations

from datetime import UTC, datetime

from narrative_dislocation.data.cache import DiskCache
from narrative_dislocation.data.universe import (
    NASDAQ100_URL,
    SP500_URL,
    UniverseLoader,
    filter_by_market_cap,
    normalize_symbol,
)

SP500_HTML = """
<html><table>
  <thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr></thead>
  <tbody>
    <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td></tr>
    <tr><td>MSFT</td><td>Microsoft</td><td>Information Technology</td><td>Systems Software</td></tr>
  </tbody>
</table></html>
"""

NASDAQ_HTML = """
<html><table>
  <thead><tr><th>Ticker</th><th>Company</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr></thead>
  <tbody>
    <tr><td>MSFT</td><td>Microsoft Corp.</td><td>Technology</td><td>Software</td></tr>
    <tr><td>NVDA</td><td>NVIDIA</td><td>Technology</td><td>Semiconductors</td></tr>
  </tbody>
</table></html>
"""


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.headers = {"Last-Modified": "Thu, 01 Jan 2026 12:00:00 GMT"}

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        del timeout
        self.calls.append(url)
        if url == SP500_URL:
            return FakeResponse(SP500_HTML)
        if url == NASDAQ100_URL:
            return FakeResponse(NASDAQ_HTML)
        raise AssertionError(f"unexpected URL: {url}")


def test_symbol_normalization_uses_yahoo_share_class_convention() -> None:
    assert normalize_symbol(" brk.b ") == "BRK-B"
    assert normalize_symbol("BF.B") == "BF-B"
    assert normalize_symbol(None) == ""


def test_universe_loads_merges_memberships_and_caches(tmp_path) -> None:
    russell = tmp_path / "russell.csv"
    russell.write_text(
        "Ticker,Company,Sector\nNVDA,NVIDIA Corporation,Technology\nXYZ,Example Inc,Industrials\n",
        encoding="utf-8",
    )
    now = datetime(2026, 1, 2, tzinfo=UTC)
    session = FakeSession()
    cache = DiskCache(tmp_path / "cache", namespace="test", now_fn=lambda: now)
    loader = UniverseLoader(
        cache,
        session=session,
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
    )

    result = loader.load(russell_source=russell)

    assert result.symbols == ("BRK-B", "MSFT", "NVDA", "XYZ")
    assert result.by_symbol()["BRK-B"].raw_symbols == ("BRK.B",)
    assert result.by_symbol()["MSFT"].memberships == ("sp500", "nasdaq100")
    assert result.by_symbol()["NVDA"].memberships == ("nasdaq100", "russell1000")
    assert len(result.sources) == 3
    assert session.calls == [SP500_URL, NASDAQ100_URL]

    second = loader.load(russell_source=russell)
    assert second.symbols == result.symbols
    assert session.calls == [SP500_URL, NASDAQ100_URL]
    assert all(source.cached for source in second.sources)


def test_market_cap_filter_does_not_estimate_missing_values(tmp_path) -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    loader = UniverseLoader(
        DiskCache(tmp_path / "cache", namespace="test", now_fn=lambda: now),
        session=FakeSession(),
        now_fn=lambda: now,
        sleep_fn=lambda _: None,
    )
    universe = loader.load(include_russell=False)

    result = filter_by_market_cap(
        universe,
        {
            "BRK.B": {"market_cap": 500_000_000_000},
            "MSFT": 5_000_000_000,  # The requested rule is strictly greater than $5B.
            "NVDA": None,
        },
    )

    assert result.universe.symbols == ("BRK-B",)
    assert result.excluded_below_minimum == ("MSFT",)
    assert result.excluded_missing_market_cap == ("NVDA",)
    assert any("not estimated" in warning for warning in result.universe.warnings)
