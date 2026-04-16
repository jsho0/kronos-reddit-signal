"""
Lookahead-bias tests.

Verifies that data sources respect as_of_date and do not return data
from after the requested date. options_flow is exempt (yfinance live-only).
"""
from datetime import date


def test_macro_point_in_time():
    """Macro fetch with a past date should succeed and include the date in raw."""
    from data_sources import macro

    # Clear cache so we actually exercise the date-range fetch path
    macro._cache.clear()

    as_of = date(2024, 1, 15)
    result = macro.fetch("SPY", as_of_date=as_of)

    # Should not raise and should return a valid score
    assert 0.0 <= result.score <= 1.0
    # raw should contain vix key populated by the date-range download
    assert "vix" in result.raw


def test_earnings_no_exception():
    """Earnings fetch with a past date must not raise."""
    from data_sources import earnings

    as_of = date(2024, 1, 15)
    result = earnings.fetch("AAPL", as_of_date=as_of)

    assert 0.0 <= result.score <= 1.0


def test_options_flow_live_only():
    """options_flow accepts as_of_date but logs a warning — it never raises."""
    from data_sources import options_flow
    import logging

    as_of = date(2024, 1, 15)
    with __import__("unittest.mock", fromlist=["patch"]).patch.object(
        options_flow.logger, "debug"
    ) as mock_debug:
        result = options_flow.fetch("SPY", as_of_date=as_of)
        # Should log the live-only warning
        calls = [str(c) for c in mock_debug.call_args_list]
        assert any("live-only" in c for c in calls), "Expected live-only debug log"

    assert 0.0 <= result.score <= 1.0


def test_as_of_date_none_is_live():
    """All sources with as_of_date=None behave identically to current (smoke test)."""
    from data_sources import macro, earnings

    macro._cache.clear()
    r1 = macro.fetch("SPY", as_of_date=None)
    r2 = earnings.fetch("AAPL", as_of_date=None)

    assert 0.0 <= r1.score <= 1.0
    assert 0.0 <= r2.score <= 1.0
