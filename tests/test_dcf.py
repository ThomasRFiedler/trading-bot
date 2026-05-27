"""
Tests for fundamental_testing/dcf.py
All tests use synthetic data — no network calls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make both packages importable
AGENT_ROOT      = Path(__file__).parent.parent
SIGNAL_TESTING  = AGENT_ROOT.parent.parent / "Git" / "stock-signal-testing"
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(SIGNAL_TESTING))

from fundamental_testing.dcf import run_dcf, _revenue_growth_rates


# ---------------------------------------------------------------------------
# Synthetic fdata builder
# ---------------------------------------------------------------------------
def _make_fdata(
    revenues=None,
    op_incomes=None,
    net_incomes=None,
    pretax_incomes=None,
    tax_provisions=None,
    operating_cash_flows=None,
    capex=None,
    shares=None,
    total_debt=0.0,
    total_cash=0.0,
    beta=1.0,
    market_cap=1e12,
    n_quarters=8,
):
    dates = pd.date_range("2023-01-01", periods=n_quarters, freq="QE")

    if revenues is None:
        revenues = [100e9 * (1 + 0.05) ** i for i in range(n_quarters)]
    if op_incomes is None:
        op_incomes = [r * 0.25 for r in revenues]
    if net_incomes is None:
        net_incomes = [r * 0.20 for r in revenues]
    if pretax_incomes is None:
        pretax_incomes = [ni / 0.79 for ni in net_incomes]
    if tax_provisions is None:
        tax_provisions = [pi * 0.21 for pi in pretax_incomes]
    if operating_cash_flows is None:
        operating_cash_flows = [r * 0.22 for r in revenues]
    if capex is None:
        capex = [r * 0.04 for r in revenues]
    if shares is None:
        shares = [15e9] * n_quarters

    income_stmt = pd.DataFrame({
        "TotalRevenue":    revenues,
        "OperatingIncome": op_incomes,
        "NetIncome":       net_incomes,
        "PretaxIncome":    pretax_incomes,
        "TaxProvision":    tax_provisions,
        "DilutedEPS":      [ni / shares[0] for ni in net_incomes],
    }, index=dates)

    balance_sheet = pd.DataFrame({
        "SharesOutstanding": shares,
        "TotalDebt":         [total_debt] * n_quarters,
        "TotalCash":         [total_cash] * n_quarters,
        "CurrentAssets":     [200e9] * n_quarters,
        "CurrentLiabilities":[80e9]  * n_quarters,
    }, index=dates)

    cash_flow = pd.DataFrame({
        "OperatingCashFlow":  operating_cash_flows,
        "CapitalExpenditure": capex,
        "FreeCashFlow":       [ocf - c for ocf, c in zip(operating_cash_flows, capex)],
    }, index=dates)

    return {
        "income_stmt":    income_stmt,
        "balance_sheet":  balance_sheet,
        "cash_flow":      cash_flow,
        "beta":           beta,
        "market_cap":     market_cap,
        "sector":         "Technology",
        "sector_etf":     "XLK",
        "insider_pct":    0.03,
        "dividend_yield": 0.0,
        "payout_ratio":   0.0,
        "trailing_pe":    25.0,
        "insider_txns":   pd.DataFrame(columns=["date", "shares", "value", "transaction_type"]),
        "daily_price":    pd.DataFrame(),
        "sector_price":   pd.DataFrame(),
        "spx_daily":      pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDCFOutputShape:
    def test_all_three_scenarios_present(self):
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=180.0)
        for key in ("low", "mid", "high"):
            assert key in result
            assert "intrinsic_value" in result[key]
            assert "upside_pct"      in result[key]
            assert "signal"          in result[key]

    def test_wacc_and_growth_in_result(self):
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=180.0)
        assert "wacc"        in result
        assert "base_growth" in result
        assert "revenue_ttm" in result


class TestDCFMonotonicity:
    def test_low_lte_mid_lte_high_intrinsic(self):
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=150.0)
        low  = result["low"]["intrinsic_value"]
        mid  = result["mid"]["intrinsic_value"]
        high = result["high"]["intrinsic_value"]
        assert low <= mid <= high, f"Expected low({low}) <= mid({mid}) <= high({high})"

    def test_high_scenario_always_most_optimistic(self):
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=150.0)
        assert result["high"]["upside_pct"] >= result["mid"]["upside_pct"]
        assert result["mid"]["upside_pct"]  >= result["low"]["upside_pct"]


class TestDCFSensitivity:
    def test_higher_beta_lowers_intrinsic_value(self):
        """Higher beta → higher WACC → lower present values."""
        fdata_low_beta  = _make_fdata(beta=0.5)
        fdata_high_beta = _make_fdata(beta=2.0)
        iv_low  = run_dcf(fdata_low_beta,  150.0)["mid"]["intrinsic_value"]
        iv_high = run_dcf(fdata_high_beta, 150.0)["mid"]["intrinsic_value"]
        assert iv_low > iv_high

    def test_net_debt_reduces_intrinsic_value(self):
        fdata_no_debt   = _make_fdata(total_debt=0,    total_cash=0)
        fdata_high_debt = _make_fdata(total_debt=100e9, total_cash=0)
        iv_no_debt   = run_dcf(fdata_no_debt,   150.0)["mid"]["intrinsic_value"]
        iv_high_debt = run_dcf(fdata_high_debt, 150.0)["mid"]["intrinsic_value"]
        assert iv_no_debt > iv_high_debt

    def test_cash_increases_intrinsic_value(self):
        # Use large enough cash ($300B) to be visible per-share (300e9 / 15e9 = $20/share)
        fdata_no_cash   = _make_fdata(total_debt=0, total_cash=0)
        fdata_high_cash = _make_fdata(total_debt=0, total_cash=300e9)
        iv_no_cash   = run_dcf(fdata_no_cash,   150.0)["mid"]["intrinsic_value"]
        iv_high_cash = run_dcf(fdata_high_cash, 150.0)["mid"]["intrinsic_value"]
        assert iv_high_cash > iv_no_cash


class TestDCFSignals:
    def test_deeply_undervalued_gives_plus_one(self):
        """Price << all intrinsic values → all scenarios bullish."""
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=1.0)  # absurdly cheap
        assert result["low"]["signal"]  == 1
        assert result["mid"]["signal"]  == 1
        assert result["high"]["signal"] == 1

    def test_deeply_overvalued_gives_minus_one(self):
        """Price >> all intrinsic values → all scenarios bearish."""
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=1e6)  # absurdly expensive
        assert result["low"]["signal"]  == -1
        assert result["mid"]["signal"]  == -1
        assert result["high"]["signal"] == -1

    def test_fair_value_price_gives_zero(self):
        """Price ≈ intrinsic value (within margin of safety) → neutral."""
        fdata = _make_fdata()
        mid_iv = run_dcf(fdata, current_price=150.0)["mid"]["intrinsic_value"]
        # Price within ±5% of intrinsic value — should be neutral
        result = run_dcf(fdata, current_price=mid_iv * 1.05)
        assert result["mid"]["signal"] == 0

    def test_upside_pct_sign_matches_under_over(self):
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=1.0)     # undervalued
        assert result["mid"]["upside_pct"] > 0

        result = run_dcf(fdata, current_price=1e6)    # overvalued
        assert result["mid"]["upside_pct"] < 0


class TestDCFEdgeCases:
    def test_zero_revenue_does_not_crash(self):
        fdata = _make_fdata(revenues=[0.0] * 8)
        result = run_dcf(fdata, current_price=100.0)
        assert isinstance(result, dict)

    def test_negative_operating_income_does_not_crash(self):
        fdata = _make_fdata(op_incomes=[-5e9] * 8)
        result = run_dcf(fdata, current_price=100.0)
        assert isinstance(result, dict)

    def test_empty_income_stmt_does_not_crash(self):
        fdata = _make_fdata()
        fdata["income_stmt"] = pd.DataFrame()
        result = run_dcf(fdata, current_price=100.0)
        assert isinstance(result, dict)
        assert result["mid"]["intrinsic_value"] == 0.0

    def test_zero_price_does_not_crash(self):
        fdata = _make_fdata()
        result = run_dcf(fdata, current_price=0.0)
        assert isinstance(result, dict)


class TestRevenueGrowthRates:
    def test_positive_growth_series(self):
        dates = pd.date_range("2022-01-01", periods=8, freq="QE")
        is_df = pd.DataFrame({
            "TotalRevenue": [100, 110, 121, 133, 146, 161, 177, 195],
        }, index=dates)
        rates = _revenue_growth_rates(is_df)
        assert all(r > 0 for r in rates)

    def test_flat_revenue_gives_near_zero_growth(self):
        dates = pd.date_range("2022-01-01", periods=8, freq="QE")
        is_df = pd.DataFrame({"TotalRevenue": [100.0] * 8}, index=dates)
        rates = _revenue_growth_rates(is_df)
        assert all(abs(r) < 0.01 for r in rates)

    def test_too_few_quarters_returns_empty(self):
        dates = pd.date_range("2022-01-01", periods=3, freq="QE")
        is_df = pd.DataFrame({"TotalRevenue": [100, 110, 121]}, index=dates)
        rates = _revenue_growth_rates(is_df)
        assert rates == []
