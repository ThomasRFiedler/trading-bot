"""
All dcc.Interval-driven callbacks.

Fast interval (30s)  → account KPIs, portfolio chart, trade history
Slow interval (5min) → strategy section (params change rarely)
"""
from __future__ import annotations

from datetime import datetime, timezone

from dash import Input, Output, no_update

import sys
from pathlib import Path
_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))
import config


def _is_market_hours() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    now_et  = (now.hour - 4) % 24
    total   = now_et * 60 + now.minute
    return 9 * 60 + 30 <= total <= 16 * 60


def register_callbacks(app) -> None:

    # ── Header status + refresh time ─────────────────────────────────────────
    @app.callback(
        Output("header-status",  "children"),
        Output("header-refresh", "children"),
        Input("interval-fast",   "n_intervals"),
    )
    def update_header(n):
        from dashboard.data.state_reader import read_state
        from dashboard.layout.header import render_status
        state = read_state()
        model = None
        try:
            from dashboard.data.registry_reader import latest_model
            m = latest_model()
            ticker = m["ticker"] if m else config.TICKER
        except Exception:
            ticker = config.TICKER

        status_children = render_status(
            connected=state["connected"],
            stale=state["stale"],
            ticker=ticker,
        )
        ts = datetime.now().strftime("%H:%M:%S")
        return status_children, f"Refreshed {ts}"

    # ── Account KPI cards ─────────────────────────────────────────────────────
    @app.callback(
        Output("account-value-value",   "children"),
        Output("account-value-sub",     "children"),
        Output("market-value-value",    "children"),
        Output("market-value-sub",      "children"),
        Output("total-gain-loss-value", "children"),
        Output("total-gain-loss-sub",   "children"),
        Output("total-gain-loss-value", "style"),
        Output("today-gain-loss-value", "children"),
        Output("today-gain-loss-sub",   "children"),
        Output("today-gain-loss-value", "style"),
        Output("buying-power-value",    "children"),
        Output("buying-power-sub",      "children"),
        Output("positions-value",       "children"),
        Output("positions-sub",         "children"),
        Input("interval-fast",          "n_intervals"),
    )
    def update_account(n):
        from dashboard.data.state_reader import read_state, read_trade_history, pnl_today
        from dashboard.layout.section_account import (
            render_account_value, render_market_value, render_total_gain,
            render_today_gain, render_buying_power, render_positions,
            KPI_VALUE_STYLE, PLACEHOLDER,
        )

        state  = read_state()
        trades = read_trade_history()

        cum_pnl = state["cumulative_pnl"]
        acct_eq = state["account_equity"]

        av_val,  av_sub  = render_account_value(acct_eq, cum_pnl)
        mv_val,  mv_sub  = render_market_value(state["in_position"], state["entry_price"])
        tg_val,  tg_sub, tg_style = render_total_gain(cum_pnl, acct_eq)
        td_val,  td_sub, td_style = render_today_gain(pnl_today(trades))
        bp_val,  bp_sub  = render_buying_power(acct_eq, cum_pnl, state["in_position"])
        pos_val, pos_sub = render_positions(state["in_position"], state["position_type"])

        return (
            av_val,  av_sub,
            mv_val,  mv_sub,
            tg_val,  tg_sub, tg_style,
            td_val,  td_sub, td_style,
            bp_val,  bp_sub,
            pos_val, pos_sub,
        )

    # ── Portfolio chart ───────────────────────────────────────────────────────
    @app.callback(
        Output("portfolio-chart", "figure"),
        Input("interval-fast",    "n_intervals"),
    )
    def update_portfolio(n):
        from dashboard.data.state_reader import read_state, read_trade_history, build_equity_series
        from dashboard.data.benchmark import get_benchmark_series
        from dashboard.layout.section_portfolio import build_figure

        state   = read_state()
        trades  = read_trade_history()
        eq_df   = build_equity_series(trades, state["account_equity"])

        align_date = eq_df["timestamp"].iloc[0] if not eq_df.empty else None
        spy        = get_benchmark_series(align_date)

        return build_figure(eq_df, spy, state["account_equity"])

    # ── Trade history (table + bar + stats) ───────────────────────────────────
    @app.callback(
        Output("pnl-bar-chart",  "figure"),
        Output("trade-stats-row","children"),
        Output("trades-table",   "children"),
        Input("interval-fast",   "n_intervals"),
    )
    def update_trades(n):
        from dashboard.data.state_reader import read_trade_history
        from dashboard.layout.section_trades import (
            render_pnl_chart, render_stats, render_table,
        )
        trades = read_trade_history()
        return render_pnl_chart(trades), render_stats(trades), render_table(trades)

    # ── Crypto price chart + signal panel ────────────────────────────────────
    @app.callback(
        Output("crypto-chart",   "figure"),
        Output("crypto-section", "style"),
        Input("interval-fast",   "n_intervals"),
    )
    def update_crypto(n):
        from dashboard.data.state_reader import read_state, read_trade_history
        from dashboard.data.price_reader import fetch_crypto_ohlc
        from dashboard.data.signal_reader import compute_bar_signals
        from dashboard.layout.section_crypto import build_figure, _empty_figure

        state = read_state()
        mode  = state.get("trading_mode") or getattr(config, "TRADING_MODE", "")

        hide = {"display": "none"}
        show = {}

        if mode != "crypto":
            return _empty_figure("Not in crypto mode"), hide

        ticker   = getattr(config, "CRYPTO_TICKER", "BTC")
        interval = getattr(config, "INTERVAL", "5m")

        # Price data for chart (2d window for display)
        price_df = fetch_crypto_ohlc(ticker=ticker, interval=interval, period="2d")
        trades   = read_trade_history()

        # Signal data — uses 5d for warm-up; build_figure trims to price window
        try:
            signals_df, signal_meta = compute_bar_signals(
                ticker=ticker, interval=interval
            )
        except Exception:
            signals_df, signal_meta = None, None

        return build_figure(
            price_df, trades, state, ticker,
            signals_df=signals_df,
            signal_meta=signal_meta,
        ), show

    # ── Strategy divergence (slow refresh) ───────────────────────────────────
    @app.callback(
        Output("divergence-status-banner", "children"),
        Output("divergence-exit-chart",    "figure"),
        Output("divergence-metrics-table", "children"),
        Output("divergence-pnl-chart",     "figure"),
        Input("interval-slow",             "n_intervals"),
    )
    def update_divergence(n):
        from dashboard.data.state_reader import read_state
        from dashboard.layout.section_divergence import (
            render_banner, render_exit_chart,
            render_metrics_table, render_pnl_chart,
            _empty_figure,
        )
        try:
            from tools.divergence_tool import compute_divergence
            state  = read_state()
            crypto = (state.get("trading_mode") or
                      getattr(config, "TRADING_MODE", "stock")) == "crypto"
            result = compute_divergence(crypto=crypto)
        except Exception as exc:
            result = {"status": "backtest_error", "error": str(exc),
                      "live_trades": 0, "backtest_trades": 0,
                      "metrics": [], "flagged_count": 0}

        return (
            render_banner(result),
            render_exit_chart(result),
            render_metrics_table(result),
            render_pnl_chart(result),
        )

    # ── Strategy section (slow refresh) ──────────────────────────────────────
    @app.callback(
        Output("strategy-params-card", "children"),
        Output("weights-chart",        "figure"),
        Output("model-history-table",  "children"),
        Input("interval-slow",         "n_intervals"),
    )
    def update_strategy(n):
        from dashboard.data.params_reader import (
            read_technical_params, read_fundamental_params,
            weights_with_names,
        )
        from dashboard.data.registry_reader import latest_model, model_history_df
        from dashboard.layout.section_strategy import (
            render_params_card, render_weights_chart, render_history_table,
        )

        # Determine active model
        model   = latest_model()
        mtype   = model["model_type"]   if model else "technical"
        ticker  = model["ticker"]       if model else config.TICKER
        dep_at  = model["deployed_at"]  if model else ""

        params  = read_technical_params()

        params_card = render_params_card(params, mtype, dep_at, ticker)
        weights     = weights_with_names(params, mtype) if params else []
        chart       = render_weights_chart(weights)
        hist_table  = render_history_table(model_history_df())

        return params_card, chart, hist_table
