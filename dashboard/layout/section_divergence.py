"""
Strategy Divergence section.

Compares live/paper trade distribution to backtest expectations and flags
metrics where real-world behaviour has diverged from the model's predictions.

Layout
------
  Status banner  (green OK / yellow warning / red alert + flagged count)
  [Exit Distribution]          [Metric Comparison Table]
   Backtest donut | Live donut   tp_rate, sl_rate, win_rate, avg_pnl …
  [P/L Distribution]
   Overlay histogram — backtest vs live
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
import dash_bootstrap_components as dbc

from .theme import (
    CARD_STYLE, SECTION_HEADER, KPI_LABEL_STYLE, PLOTLY_LAYOUT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW, ACCENT_TEAL, ACCENT_BLUE,
    TEXT_PRIMARY, TEXT_MUTED, TEXT_DIM,
    BG_CARD, BG_CARD_ALT, BG_HEADER, BORDER,
)

_STATUS_COLORS = {
    "ok":                ACCENT_GREEN,
    "warning":           ACCENT_YELLOW,
    "alert":             ACCENT_RED,
    "insufficient_data": TEXT_MUTED,
    "no_params":         TEXT_MUTED,
    "backtest_error":    ACCENT_RED,
}
_STATUS_LABELS = {
    "ok":                "Within expected range",
    "warning":           "Minor divergence detected",
    "alert":             "Significant divergence — review parameters",
    "insufficient_data": "Collecting live trades…",
    "no_params":         "No deployed parameters found",
    "backtest_error":    "Backtest failed — check logs",
}

_EXIT_COLORS = {
    "take_profit": ACCENT_GREEN,
    "stop_loss":   ACCENT_RED,
    "eod_exit":    ACCENT_BLUE,
}

_METRIC_LABELS = {
    "tp_rate":        "Take-Profit Rate",
    "sl_rate":        "Stop-Loss Rate",
    "eod_rate":       "EOD Exit Rate",
    "win_rate":       "Win Rate",
    "avg_net_pnl":    "Avg Net P/L",
    "avg_hold_bars":  "Avg Hold (bars)",
    "trade_freq_day": "Trades / Day",
    "profit_factor":  "Profit Factor",
    "sharpe_ratio":   "Sharpe Ratio",
}

_PP_METRICS = {"tp_rate", "sl_rate", "eod_rate", "win_rate"}


def build_section() -> html.Div:
    return html.Div([
        html.P("Strategy Divergence", style=SECTION_HEADER),
        html.Div(id="divergence-status-banner",
                 style={**CARD_STYLE, "padding": "12px 20px"}),
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.P("Exit Reason Distribution",
                           style={**KPI_LABEL_STYLE, "marginBottom": "8px"}),
                    dcc.Graph(
                        id="divergence-exit-chart",
                        figure=_empty_figure(""),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "260px"},
                    ),
                ]),
            ], md=5),
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.P("Metric Comparison",
                           style={**KPI_LABEL_STYLE, "marginBottom": "8px"}),
                    html.Div(id="divergence-metrics-table"),
                ]),
            ], md=7),
        ], className="g-2"),
        html.Div(style=CARD_STYLE, children=[
            html.P("P/L Distribution — Backtest vs Live",
                   style={**KPI_LABEL_STYLE, "marginBottom": "8px"}),
            dcc.Graph(
                id="divergence-pnl-chart",
                figure=_empty_figure(""),
                config={"displayModeBar": False, "responsive": True},
                style={"height": "200px"},
            ),
        ]),
    ])


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def render_banner(result: dict) -> html.Div:
    status  = result.get("status", "insufficient_data")
    color   = _STATUS_COLORS.get(status, TEXT_MUTED)
    label   = _STATUS_LABELS.get(status, status)
    flagged = result.get("flagged_count", 0)
    live_n  = result.get("live_trades", 0)
    bt_n    = result.get("backtest_trades", 0)
    min_req = result.get("min_required", 5)

    if status == "insufficient_data":
        sub = f"{live_n} / {min_req} live trades collected (need {min_req - live_n} more)"
    elif status in ("no_params", "backtest_error"):
        sub = result.get("error", "")
    else:
        total  = len(result.get("metrics", []))
        sub    = (f"{flagged} / {total} metrics flagged  ·  "
                  f"{live_n} live trades vs {bt_n} backtest trades")

    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "12px"},
        children=[
            html.Div(
                style={"width": "10px", "height": "10px", "borderRadius": "50%",
                       "backgroundColor": color, "flexShrink": "0"},
            ),
            html.Div([
                html.Span(label, style={"color": color,
                                        "fontWeight": "600", "fontSize": "13px"}),
                html.Br(),
                html.Span(sub,   style={"color": TEXT_MUTED, "fontSize": "12px"}),
            ]),
        ],
    )


# ---------------------------------------------------------------------------
# Exit-reason donuts
# ---------------------------------------------------------------------------

def render_exit_chart(result: dict) -> go.Figure:
    bt_counts   = result.get("bt_exit_counts",   {})
    live_counts = result.get("live_exit_counts",  {})

    if not bt_counts and not live_counts:
        return _empty_figure("No exit data")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Backtest", "Live / Paper"],
    )

    for col, counts, title in [(1, bt_counts, "Backtest"),
                                (2, live_counts, "Live")]:
        if not counts or sum(counts.values()) == 0:
            fig.add_trace(
                go.Pie(
                    labels=["No data"],
                    values=[1],
                    marker=dict(colors=[BG_CARD_ALT]),
                    showlegend=False,
                    textinfo="label",
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )
            continue

        labels = [k.replace("_", " ").title() for k in counts]
        values = list(counts.values())
        colors = [_EXIT_COLORS.get(k, TEXT_MUTED) for k in counts]

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color=BG_CARD, width=2)),
                hole=0.55,
                textinfo="percent",
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="v", x=1.02, y=0.5,
                    font=dict(size=11, color=TEXT_MUTED)),
        annotations=[
            dict(text=a["text"],
                 x=a["x"], y=0.5,
                 xref="paper", yref="paper",
                 font=dict(size=11, color=TEXT_MUTED),
                 showarrow=False)
            for a in fig.layout.annotations
        ],
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Metrics comparison table
# ---------------------------------------------------------------------------

def render_metrics_table(result: dict) -> html.Div:
    metrics = result.get("metrics", [])
    if not metrics:
        return html.Div("Waiting for data…",
                        style={"color": TEXT_MUTED, "fontSize": "13px"})

    rows = []
    for m in metrics:
        key      = m["metric"]
        label    = _METRIC_LABELS.get(key, key)
        bt_val   = m["backtest"]
        live_val = m["live"]
        diff     = m["diff"]
        flagged  = m["flagged"]

        is_pp = key in _PP_METRICS

        def fmt(v):
            if key in ("tp_rate", "sl_rate", "eod_rate", "win_rate"):
                return f"{v:.1f}%"
            if key == "avg_net_pnl":
                return f"${v:+.2f}"
            if key == "avg_hold_bars":
                return f"{v:.1f}"
            if key == "trade_freq_day":
                return f"{v:.1f}/day"
            if key == "profit_factor":
                return "∞" if v == float("inf") else f"{v:.2f}x"
            if key == "sharpe_ratio":
                return f"{v:.3f}"
            return str(round(v, 2))

        diff_str = (f"{diff:+.1f}pp" if is_pp
                    else f"{diff:+.1f}%" if m["metric"] not in ("sharpe_ratio",)
                    else f"{diff:+.3f}")
        diff_color = (ACCENT_GREEN if diff >= 0 else ACCENT_RED) if not flagged else ACCENT_RED

        row_bg   = "rgba(248,81,73,0.08)" if flagged else "transparent"
        flag_icon = html.Span(" ⚠", style={"color": ACCENT_YELLOW}) if flagged else ""

        rows.append(
            html.Tr([
                html.Td([label, flag_icon],
                        style={"color": TEXT_PRIMARY, "padding": "5px 8px",
                               "fontSize": "12px"}),
                html.Td(fmt(bt_val),
                        style={"color": TEXT_MUTED,    "padding": "5px 8px",
                               "fontSize": "12px", "textAlign": "right",
                               "fontFamily": "monospace"}),
                html.Td(fmt(live_val),
                        style={"color": TEXT_PRIMARY,  "padding": "5px 8px",
                               "fontSize": "12px", "textAlign": "right",
                               "fontFamily": "monospace",
                               "fontWeight": "600" if flagged else "400"}),
                html.Td(diff_str,
                        style={"color": diff_color,    "padding": "5px 8px",
                               "fontSize": "12px", "textAlign": "right",
                               "fontFamily": "monospace"}),
            ], style={"backgroundColor": row_bg})
        )

    return html.Table(
        [
            html.Thead(html.Tr([
                html.Th("Metric",   style={"color": TEXT_MUTED, "fontSize": "11px",
                                           "padding": "4px 8px", "fontWeight": "600"}),
                html.Th("Backtest", style={"color": TEXT_MUTED, "fontSize": "11px",
                                           "padding": "4px 8px", "textAlign": "right"}),
                html.Th("Live",     style={"color": TEXT_MUTED, "fontSize": "11px",
                                           "padding": "4px 8px", "textAlign": "right"}),
                html.Th("Δ",        style={"color": TEXT_MUTED, "fontSize": "11px",
                                           "padding": "4px 8px", "textAlign": "right"}),
            ])),
            html.Tbody(rows),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


# ---------------------------------------------------------------------------
# P/L distribution overlay histogram
# ---------------------------------------------------------------------------

def render_pnl_chart(result: dict) -> go.Figure:
    bt_pnl   = result.get("bt_pnl_dist",   [])
    live_pnl = result.get("live_pnl_dist",  [])

    if not bt_pnl and not live_pnl:
        return _empty_figure("No P/L data")

    fig = go.Figure()

    if bt_pnl:
        fig.add_trace(go.Histogram(
            x=bt_pnl,
            name="Backtest",
            marker_color=ACCENT_TEAL,
            opacity=0.55,
            nbinsx=30,
            hovertemplate="P/L: %{x:.2f}<br>Count: %{y}<extra>Backtest</extra>",
        ))

    if live_pnl:
        fig.add_trace(go.Histogram(
            x=live_pnl,
            name="Live",
            marker_color=ACCENT_YELLOW,
            opacity=0.7,
            nbinsx=max(10, len(live_pnl)),
            hovertemplate="P/L: %{x:.2f}<br>Count: %{y}<extra>Live</extra>",
        ))

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        barmode="overlay",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", x=0, y=1.1,
                    font=dict(size=11, color=TEXT_MUTED)),
        xaxis=dict(title=dict(text="Net P/L ($)", font=dict(size=11)),
                   showgrid=True, gridcolor=BORDER, zeroline=True,
                   zerolinecolor=BORDER),
        yaxis=dict(title=dict(text="Trades", font=dict(size=11)),
                   showgrid=True, gridcolor=BORDER),
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    if msg:
        fig.add_annotation(
            text=msg, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=TEXT_MUTED),
        )
    fig.update_layout(**PLOTLY_LAYOUT, height=180)
    return fig
