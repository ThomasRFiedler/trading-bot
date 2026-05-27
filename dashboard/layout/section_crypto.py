"""
Crypto price chart section.

Shows a three-row chart:
  Row 1 — Candlestick with trade entry/exit markers and open-position line
  Row 2 — Signal sum panel: weighted indicator sum vs entry threshold ±n
  Row 3 — Volume bars

Only renders when trading_mode == "crypto".
"""
from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html

from .theme import (
    CARD_STYLE, SECTION_HEADER, PLOTLY_LAYOUT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_BLUE, ACCENT_YELLOW,
    TEXT_MUTED, TEXT_DIM, BG_CARD, BORDER,
)

_RANGE_BUTTONS = [
    dict(count=6,  label="6H",  step="hour", stepmode="backward"),
    dict(count=1,  label="1D",  step="day",  stepmode="backward"),
    dict(count=3,  label="3D",  step="day",  stepmode="backward"),
    dict(             label="All", step="all"),
]

# Signal panel colours
_SIG_LINE   = "#7ab8f5"   # neutral blue line
_SIG_LONG   = "#1f4e2a"   # dark-green fill above +n
_SIG_SHORT  = "#4e1f1f"   # dark-red fill below -n
_SIG_THRESH = ACCENT_YELLOW


def build_section() -> html.Div:
    return html.Div(
        id="crypto-section",
        children=[
            html.P("Crypto Price & Signals", style=SECTION_HEADER),
            html.Div(
                style=CARD_STYLE,
                children=[
                    dcc.Graph(
                        id="crypto-chart",
                        figure=_empty_figure("Waiting for price data…"),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "540px"},
                    ),
                ],
            ),
        ],
    )


def build_figure(
    price_df: pd.DataFrame,
    trades: list[dict],
    state: dict,
    ticker: str = "BTC",
    signals_df: pd.DataFrame | None = None,
    signal_meta: dict | None = None,
) -> go.Figure:
    """
    Build the three-panel chart.

    price_df    : OHLCV DataFrame (tz-aware index)
    trades      : from state_reader.read_trade_history()
    state       : from state_reader.read_state()
    ticker      : crypto symbol
    signals_df  : per-bar signal data from signal_reader.compute_bar_signals()
    signal_meta : meta dict from compute_bar_signals() (contains n, ind_names, etc.)
    """
    if price_df.empty:
        return _empty_figure(f"No price data for {ticker}-USD")

    have_signals = (
        signals_df is not None
        and not signals_df.empty
        and signal_meta is not None
        and signal_meta.get("params_loaded")
    )

    # ── Subplots ──────────────────────────────────────────────────────────────
    if have_signals:
        row_heights    = [0.60, 0.25, 0.15]
        subplot_titles = ("", "", "")
    else:
        row_heights    = [0.80, 0.20]
        subplot_titles = ("", "")

    n_rows = 3 if have_signals else 2
    fig    = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )

    vol_row    = n_rows          # volume is always the last row
    signal_row = 2 if have_signals else None

    # ── Row 1: Candlesticks ───────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            name=f"{ticker}-USD",
            increasing=dict(line=dict(color=ACCENT_GREEN), fillcolor=ACCENT_GREEN),
            decreasing=dict(line=dict(color=ACCENT_RED),   fillcolor=ACCENT_RED),
            hoverinfo="x+y",
        ),
        row=1, col=1,
    )

    # ── Row 2: Signal sum panel ───────────────────────────────────────────────
    if have_signals:
        n         = float(signal_meta["n"])
        ind_names = signal_meta.get("ind_names", [])
        weights   = signal_meta.get("weights", [])

        # Trim signals to match the visible price window
        sig = signals_df.loc[signals_df.index >= price_df.index[0]].copy()

        if not sig.empty:
            sums = sig["signal_sum"].values
            idx  = sig.index

            # Background fills: long zone (above +n) and short zone (below -n)
            fig.add_hrect(
                y0=n, y1=max(float(np.nanmax(sums)) * 1.1, n * 2),
                fillcolor=_SIG_LONG, opacity=0.4, line_width=0,
                row=signal_row, col=1,
            )
            fig.add_hrect(
                y0=min(float(np.nanmin(sums)) * 1.1, -n * 2), y1=-n,
                fillcolor=_SIG_SHORT, opacity=0.4, line_width=0,
                row=signal_row, col=1,
            )

            # Build hover text with per-indicator breakdown
            hover_parts = []
            for k, s_val in enumerate(sums):
                parts = [f"<b>sum={s_val:+.3f}</b>  n=±{n:.3f}"]
                for name, w in zip(ind_names, weights):
                    col_name = f"ind_{name}"
                    if col_name in sig.columns and w != 0.0:
                        raw    = int(sig[col_name].iloc[k])
                        contrib = raw * w
                        arrow  = "▲" if raw > 0 else ("▼" if raw < 0 else "·")
                        parts.append(f"{arrow} {name}: {contrib:+.2f}")
                hover_parts.append("<br>".join(parts))

            # Signal sum line
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=sums,
                    mode="lines",
                    name="Signal",
                    line=dict(color=_SIG_LINE, width=1.5),
                    hovertext=hover_parts,
                    hovertemplate="%{hovertext}<extra></extra>",
                ),
                row=signal_row, col=1,
            )

            # Threshold lines at ±n
            fig.add_hline(
                y=n, line_dash="dot", line_color=_SIG_THRESH, line_width=1,
                annotation_text=f"  +n={n:.2f}",
                annotation_font=dict(size=10, color=_SIG_THRESH),
                annotation_position="top right",
                row=signal_row, col=1,
            )
            fig.add_hline(
                y=-n, line_dash="dot", line_color=_SIG_THRESH, line_width=1,
                annotation_text=f"  -n={-n:.2f}",
                annotation_font=dict(size=10, color=_SIG_THRESH),
                annotation_position="bottom right",
                row=signal_row, col=1,
            )

            # Current bar: large marker coloured by direction
            last_sum = float(sums[-1])
            if last_sum >= n:
                dot_color, dot_sym, status = ACCENT_GREEN, "triangle-up",   "LONG"
            elif last_sum <= -n:
                dot_color, dot_sym, status = ACCENT_RED,   "triangle-down", "SHORT"
            else:
                dot_color, dot_sym, status = _SIG_LINE,    "circle",        "flat"

            fig.add_trace(
                go.Scatter(
                    x=[idx[-1]],
                    y=[last_sum],
                    mode="markers+text",
                    name="Now",
                    marker=dict(symbol=dot_sym, size=10, color=dot_color,
                                line=dict(width=1.5, color="white")),
                    text=[f"  {last_sum:+.2f} ({status})"],
                    textposition="middle right",
                    textfont=dict(size=11, color=dot_color),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=signal_row, col=1,
            )

    # ── Volume row ────────────────────────────────────────────────────────────
    vol_colors = [
        ACCENT_GREEN if c >= o else ACCENT_RED
        for o, c in zip(price_df["open"], price_df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=price_df.index,
            y=price_df["volume"],
            name="Volume",
            marker_color=vol_colors,
            marker_opacity=0.5,
            showlegend=False,
            hovertemplate="%{y:,.0f}<extra>Vol</extra>",
        ),
        row=vol_row, col=1,
    )

    # ── Trade entry / exit markers on Row 1 ───────────────────────────────────
    cutoff = price_df.index.min()
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")

    crypto_trades = [
        t for t in trades
        if t.get("trading_mode") == "crypto" or t.get("ticker") == ticker
    ]

    entry_long_x, entry_long_y   = [], []
    entry_short_x, entry_short_y = [], []
    exit_x, exit_y, exit_text, exit_colors = [], [], [], []

    for t in crypto_trades:
        ts = t["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff.to_pydatetime():
            continue

        entry_ts_str = t.get("entry_time", "")
        try:
            ets = pd.Timestamp(entry_ts_str)
            if ets.tzinfo is None:
                ets = ets.tz_localize("UTC")
        except Exception:
            ets = ts

        ep = t.get("entry_price", 0)
        if t["direction"] == "LONG":
            entry_long_x.append(ets)
            entry_long_y.append(ep * 0.9995)
        else:
            entry_short_x.append(ets)
            entry_short_y.append(ep * 1.0005)

        pnl   = t["net_pnl"]
        sign  = "+" if pnl >= 0 else ""
        color = ACCENT_GREEN if pnl >= 0 else ACCENT_RED
        exit_x.append(ts)
        exit_y.append(t["exit_price"])
        exit_text.append(f"{sign}${pnl:.2f}")
        exit_colors.append(color)

    if entry_long_x:
        fig.add_trace(
            go.Scatter(
                x=entry_long_x, y=entry_long_y,
                mode="markers", name="Entry Long",
                marker=dict(symbol="triangle-up", size=12,
                            color=ACCENT_GREEN, line=dict(width=1, color="white")),
                hovertemplate="Entry Long<br>%{x|%H:%M}<br>%{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if entry_short_x:
        fig.add_trace(
            go.Scatter(
                x=entry_short_x, y=entry_short_y,
                mode="markers", name="Entry Short",
                marker=dict(symbol="triangle-down", size=12,
                            color=ACCENT_RED, line=dict(width=1, color="white")),
                hovertemplate="Entry Short<br>%{x|%H:%M}<br>%{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if exit_x:
        fig.add_trace(
            go.Scatter(
                x=exit_x, y=exit_y,
                mode="markers+text", name="Exit",
                marker=dict(symbol="x", size=10,
                            color=exit_colors, line=dict(width=2)),
                text=exit_text,
                textposition="top center",
                textfont=dict(size=10, color=exit_colors),
                hovertemplate="Exit<br>%{x|%H:%M}<br>%{y:,.2f}<br>%{text}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Open position entry price line ────────────────────────────────────────
    if state.get("in_position") and state.get("entry_price"):
        ep    = state["entry_price"]
        color = ACCENT_GREEN if state.get("position_type") == "long" else ACCENT_RED
        fig.add_hline(
            y=ep, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=f"  Entry {ep:,.2f}",
            annotation_font=dict(size=11, color=color),
            annotation_position="top right",
            row=1, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    last_price = price_df["close"].iloc[-1]
    prev_close = price_df["close"].iloc[-2] if len(price_df) > 1 else last_price
    change_pct = (last_price - prev_close) / prev_close * 100
    sign       = "+" if change_pct >= 0 else ""
    tc         = ACCENT_GREEN if change_pct >= 0 else ACCENT_RED

    # Signal status suffix for title
    # `sig` may not be defined if have_signals is False or sig was empty
    _sig_for_title = (
        signals_df.loc[signals_df.index >= price_df.index[0]]
        if have_signals and signals_df is not None and not signals_df.empty
        else pd.DataFrame()
    )
    if have_signals and not _sig_for_title.empty:
        last_sum  = float(_sig_for_title["signal_sum"].iloc[-1])
        if last_sum >= n:
            sig_label = f"  ·  <span style='color:{ACCENT_GREEN}'>▲ LONG SIGNAL</span>"
        elif last_sum <= -n:
            sig_label = f"  ·  <span style='color:{ACCENT_RED}'>▼ SHORT SIGNAL</span>"
        else:
            sig_label = (
                f"  ·  <span style='color:{TEXT_MUTED}'>"
                f"flat  ({last_sum:+.2f} / ±{n:.2f})</span>"
            )
    else:
        sig_label = ""

    layout_overrides = {
        k: v for k, v in PLOTLY_LAYOUT.items()
        if k not in ("xaxis", "yaxis")
    }
    fig.update_layout(
        **layout_overrides,
        height=540 if have_signals else 420,
        title=dict(
            text=(
                f"{ticker}-USD  ·  "
                f"<span style='color:{tc}'>${last_price:,.2f}  {sign}{change_pct:.2f}%</span>"
                f"{sig_label}"
            ),
            font=dict(size=13, color=TEXT_MUTED),
            x=0, pad=dict(l=4),
        ),
        # Price axis
        xaxis=dict(
            showgrid=True, gridcolor=BORDER, zeroline=False,
            linecolor=BORDER, tickcolor=TEXT_DIM,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=_RANGE_BUTTONS,
                bgcolor=BG_CARD, activecolor=ACCENT_BLUE,
                bordercolor=BORDER,
                font=dict(color=TEXT_MUTED, size=11),
                x=0, y=1.06,
            ),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=BORDER, zeroline=False,
            linecolor=BORDER, tickcolor=TEXT_DIM,
            tickformat=",.0f",
            title=dict(text="Price (USD)", font=dict(size=11, color=TEXT_MUTED)),
        ),
        legend=dict(
            orientation="h", x=0, y=-0.04,
            font=dict(size=11, color=TEXT_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    # Signal axis — zero line, no grid clutter
    if have_signals:
        sig_axis_key = f"yaxis{signal_row}"
        fig.update_layout(**{
            sig_axis_key: dict(
                showgrid=True, gridcolor=BORDER, zeroline=True,
                zerolinecolor=BORDER, zerolinewidth=1,
                linecolor=BORDER, tickcolor=TEXT_DIM,
                tickformat=".2f",
                title=dict(text="Signal", font=dict(size=10, color=TEXT_MUTED)),
            ),
        })

    # Volume axis
    vol_axis_key = f"yaxis{vol_row}"
    fig.update_layout(**{
        vol_axis_key: dict(
            showgrid=False, zeroline=False,
            linecolor=BORDER, tickcolor=TEXT_DIM,
            tickformat=".2s",
            title=dict(text="Volume", font=dict(size=10, color=TEXT_MUTED)),
        ),
        f"xaxis{vol_row}": dict(
            showgrid=True, gridcolor=BORDER, zeroline=False,
            linecolor=BORDER, tickcolor=TEXT_DIM,
        ),
    })

    return fig


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color=TEXT_MUTED),
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    return fig
