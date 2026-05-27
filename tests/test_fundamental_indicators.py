"""
Tests for fundamental_testing/indicators.py
All tests use synthetic data — no network calls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AGENT_ROOT     = Path(__file__).parent.parent
SIGNAL_TESTING = AGENT_ROOT.parent.parent / "Git" / "stock-signal-testing"
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(SIGNAL_TESTING))

from fundamental_testing.indicators import (
    INDICATORS,
    N_INDICATORS,
    _VEC_FUNCTIONS,
    compute_fundamental_matrix,
    _quarterly_broadcast,
    _slope_sign,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_price_df(n_days=252, start="2023-01-02", start_price=150.0):
    dates  = pd.bdate_range(start, periods=n_days)
    prices = start_price + np.cumsum(np.random.randn(n_days) * 0.5)
    prices = np.maximum(prices, 10.0)
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.005,
        "low":    prices * 0.995,
        "close":  prices,
        "volume": np.ones(n_days) * 1e7,
    }, index=dates)


def _make_fdata(n_quarters=8, price_df=None):
    """Build synthetic fdata with enough quarters to exercise all indicators."""
    if price_df is None:
        price_df = _make_price_df()

    dates = pd.date_range("2021-01-01", periods=n_quarters, freq="QE")

    revenues     = [100e9 * (1.05 ** i) for i in range(n_quarters)]
    op_incomes   = [r * 0.25   for r in revenues]
    net_incomes  = [r * 0.20   for r in revenues]
    eps_vals     = [ni / 15e9  for ni in net_incomes]
    shares       = [15e9 * (1 - 0.01 * i) for i in range(n_quarters)]

    income_stmt = pd.DataFrame({
        "TotalRevenue":    revenues,
        "OperatingIncome": op_incomes,
        "NetIncome":       net_incomes,
        "DilutedEPS":      eps_vals,
        "PretaxIncome":    [ni / 0.79 for ni in net_incomes],
        "TaxProvision":    [ni * 0.21 / 0.79 for ni in net_incomes],
    }, index=dates)

    balance_sheet = pd.DataFrame({
        "SharesOutstanding":  shares,
        "CurrentAssets":      [200e9] * n_quarters,
        "CurrentLiabilities": [80e9]  * n_quarters,
        "TotalDebt":          [50e9]  * n_quarters,
        "TotalCash":          [30e9]  * n_quarters,
    }, index=dates)

    cash_flow = pd.DataFrame({
        "OperatingCashFlow":  [r * 0.22 for r in revenues],
        "CapitalExpenditure": [r * 0.04 for r in revenues],
        "FreeCashFlow":       [r * 0.18 for r in revenues],
    }, index=dates)

    insider_txns = pd.DataFrame({
        "date":             pd.date_range("2023-01-01", periods=3, freq="ME"),
        "shares":           [10000, 5000, 3000],
        "value":            [1_800_000, 900_000, 540_000],
        "transaction_type": ["Buy", "Buy", "Sell"],
    })

    sector_dates  = price_df.index
    sector_prices = 450.0 + np.cumsum(np.random.randn(len(sector_dates)) * 0.5)
    sector_df = pd.DataFrame({
        "close": sector_prices,
        "open": sector_prices,
        "high": sector_prices * 1.005,
        "low":  sector_prices * 0.995,
    }, index=sector_dates)

    spx_prices = 4500.0 + np.cumsum(np.random.randn(len(sector_dates)) * 2.0)
    spx_df = pd.DataFrame({
        "close": spx_prices,
        "open":  spx_prices,
        "high":  spx_prices * 1.003,
        "low":   spx_prices * 0.997,
    }, index=sector_dates)

    return {
        "income_stmt":    income_stmt,
        "balance_sheet":  balance_sheet,
        "cash_flow":      cash_flow,
        "daily_price":    price_df,
        "sector_price":   sector_df,
        "spx_daily":      spx_df,
        "market_cap":     2.5e12,
        "sector":         "Technology",
        "sector_etf":     "XLK",
        "beta":           1.2,
        "insider_pct":    0.03,
        "dividend_yield": 0.005,
        "payout_ratio":   0.15,
        "trailing_pe":    28.0,
        "insider_txns":   insider_txns,
    }


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_n_indicators_equals_15(self):
        assert N_INDICATORS == 15

    def test_indicators_list_length(self):
        assert len(INDICATORS) == 15

    def test_vec_functions_length_matches(self):
        assert len(_VEC_FUNCTIONS) == N_INDICATORS

    def test_all_indicator_names_unique(self):
        names = [name for name, _ in INDICATORS]
        assert len(names) == len(set(names))

    def test_default_weights_nonnegative(self):
        for name, w in INDICATORS:
            assert w >= 0, f"{name} has negative default weight"


# ---------------------------------------------------------------------------
# compute_fundamental_matrix output shape and dtype
# ---------------------------------------------------------------------------
class TestComputeFundamentalMatrix:
    def test_output_shape(self):
        price_df = _make_price_df(n_days=252)
        fdata    = _make_fdata(price_df=price_df)
        matrix   = compute_fundamental_matrix(price_df, fdata)
        assert matrix.shape == (252, 15)

    def test_output_dtype_int8(self):
        price_df = _make_price_df(n_days=100)
        fdata    = _make_fdata(price_df=price_df)
        matrix   = compute_fundamental_matrix(price_df, fdata)
        assert matrix.dtype == np.int8

    def test_values_only_in_minus1_0_plus1(self):
        price_df = _make_price_df(n_days=100)
        fdata    = _make_fdata(price_df=price_df)
        matrix   = compute_fundamental_matrix(price_df, fdata)
        unique_vals = set(np.unique(matrix))
        assert unique_vals.issubset({-1, 0, 1}), f"Unexpected values: {unique_vals}"

    def test_no_nan_in_matrix(self):
        price_df = _make_price_df(n_days=100)
        fdata    = _make_fdata(price_df=price_df)
        matrix   = compute_fundamental_matrix(price_df, fdata)
        assert not np.any(np.isnan(matrix.astype(float)))


# ---------------------------------------------------------------------------
# Individual vectorized functions — each returns correct shape/dtype
# ---------------------------------------------------------------------------
class TestIndividualVecFunctions:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.price_df = _make_price_df(n_days=252)
        self.fdata    = _make_fdata(price_df=self.price_df)
        self.n        = len(self.price_df)

    def _check(self, fn):
        result = fn(self.price_df, self.fdata)
        assert result.shape == (self.n,), f"{fn.__name__}: shape {result.shape} != ({self.n},)"
        assert result.dtype == np.int8,   f"{fn.__name__}: dtype {result.dtype} != int8"
        assert set(np.unique(result)).issubset({-1, 0, 1}), \
            f"{fn.__name__}: values not in {{-1,0,1}}: {np.unique(result)}"
        return result

    def test_all_vec_functions(self):
        for fn in _VEC_FUNCTIONS:
            self._check(fn)

    def test_empty_price_df_returns_zeros(self):
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        for fn in _VEC_FUNCTIONS:
            result = fn(empty_df, self.fdata)
            assert len(result) == 0 or np.all(result == 0)


# ---------------------------------------------------------------------------
# Quarterly broadcast correctness
# ---------------------------------------------------------------------------
class TestQuarterlyBroadcast:
    def test_broadcast_holds_value_within_quarter(self):
        dates  = pd.bdate_range("2023-01-02", periods=130)
        price_df = pd.DataFrame({"close": np.ones(130)}, index=dates)

        # One quarterly signal at the start of the year
        q_signal = pd.Series({pd.Timestamp("2023-03-31"): np.int8(1)})
        result   = _quarterly_broadcast(price_df, q_signal)

        # All bars on or after 2023-03-31 should be 1
        for i, d in enumerate(dates):
            if d >= pd.Timestamp("2023-03-31"):
                assert result[i] == 1, f"Expected 1 at {d}, got {result[i]}"
            else:
                assert result[i] == 0, f"Expected 0 at {d}, got {result[i]}"

    def test_broadcast_updates_on_new_quarter(self):
        dates  = pd.bdate_range("2023-01-02", periods=200)
        price_df = pd.DataFrame({"close": np.ones(200)}, index=dates)

        q_signal = pd.Series({
            pd.Timestamp("2023-03-31"): np.int8(1),
            pd.Timestamp("2023-06-30"): np.int8(-1),
        })
        result = _quarterly_broadcast(price_df, q_signal)

        q1_end = pd.Timestamp("2023-06-30")
        for i, d in enumerate(dates):
            if d >= q1_end:
                assert result[i] == -1
            elif d >= pd.Timestamp("2023-03-31"):
                assert result[i] == 1

    def test_empty_quarterly_signal_returns_zeros(self):
        price_df = _make_price_df(100)
        result   = _quarterly_broadcast(price_df, pd.Series(dtype=np.int8))
        assert np.all(result == 0)


# ---------------------------------------------------------------------------
# Slope sign helper
# ---------------------------------------------------------------------------
class TestSlopeSign:
    def test_increasing_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert _slope_sign(s) == 1

    def test_decreasing_series(self):
        s = pd.Series([4.0, 3.0, 2.0, 1.0])
        assert _slope_sign(s) == -1

    def test_flat_series_returns_zero(self):
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        assert _slope_sign(s) == 0

    def test_too_short_returns_zero(self):
        s = pd.Series([1.0, 2.0])
        assert _slope_sign(s, min_points=3) == 0
