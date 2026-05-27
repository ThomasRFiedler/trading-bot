"""Trade history: table + per-trade PnL bar chart."""
from __future__ import annotations

import plotly.graph_objects as go
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc

from .theme import (
    CARD_STYLE, SECTION_HEADER, KPI_LABEL_STYLE, PLOTLY_LAYOUT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_TEAL, TEXT_PRIMARY, TEXT_MUTED,
    BG_CARD, BG_HEADER, BORDER, TABLE_STYLE_HEADER, TABLE_STYLE_CELL,
)


def build_section() -> html.Div:
    return html.Div([
        html.P("Trade History", style=SECTION_HEADER),
        dbc.Row([
            # PnL bar chart
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.P("Per-Trade P&L", style={**KPI_LABEL_STYLE, "marginBottom": "8px"}),
                    dcc.Graph(
                        id="pnl-bar-chart",
                        figure=_empty_chart("No trades yet"),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "240px"},
                    ),
                ]),
            ], md=5),

            # Summary stats
            dbc.Col([
                html.Div(id="trade-stats-row",
                         style={**CARD_STYLE, "display": "flex", "gap": "24px",
                                "flexWrap": "wrap", "alignItems": "center"},
                         children=_stats_placeholder()),
            ], md=7),
        ], className="g-2"),

        # Trade table
        html.Div(style=CARD_STYLE, children=[
            html.P("All Trades", style={**KPI_LABEL_STYLE, "marginBottom": "8px"}),
            html.Div(id="trades-table", children=_table_placeholder()),
        ]),
    ])


def _stats_placeholder():
    return [html.P("No trade data", style={"color": TEXT_MUTED, "fontSize": "13px"})]


def _table_placeholder():
    return [html.P("No trades recorded yet.", style={"color": TEXT_MUTED, "fontSize": "13px"})]


# ── Render helpers ────────────────────────────────────────────────────────────

def render_pnl_chart(trades: list[dict]) -> go.Figure:
    if not trades:
        return _empty_chart("No trades yet")

    pnls   = [t["net_pnl"] for t in trades]
    labels = [f"#{i+1}" for i in range(len(pnls))]
    colors = [ACCENT_TEAL if p > 0 else ACCENT_RED for p in pnls]
    hover  = [
        f"Trade #{i+1}<br>{t['direction']}<br>"
        f"{t['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>"
        f"PnL: ${t['net_pnl']:+.2f}<br>"
        f"Reason: {t['exit_reason']}"
        for i, t in enumerate(trades)
    ]

    fig = go.Figure(go.Bar(
        x             = labels,
        y             = pnls,
        marker_color  = colors,
        hovertext     = hover,
        hoverinfo     = "text",
        hovertemplate = "%{hovertext}<extra></extra>",
    ))

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        xaxis  = dict(showgrid=False, zeroline=False,
                      tickfont=dict(size=10), title=None),
        yaxis  = dict(showgrid=True, gridcolor=BORDER, zeroline=True,
                      zerolinecolor="rgba(139,148,158,0.4)", title="P&L ($)"),
        margin = dict(l=10, r=10, t=10, b=30),
        hovermode="closest",
    )
    fig.add_hline(y=0, line_color="rgba(139,148,158,0.3)", line_width=1)
    fig.update_layout(**layout)
    return fig


def render_stats(trades: list[dict]) -> list:
    if not trades:
        return _stats_placeholder()

    total   = len(trades)
    wins    = sum(1 for t in trades if t["net_pnl"] > 0)
    win_pct = wins / total * 100 if total else 0
    gross_w = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_l = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    pf      = gross_w / gross_l if gross_l > 0 else float("inf")
    avg_pnl = sum(t["net_pnl"] for t in trades) / total if total else 0

    def stat(label, value, color=TEXT_PRIMARY):
        return html.Div([
            html.Div(value, style={"color": color, "fontSize": "20px",
                                   "fontWeight": "600", "fontFamily": "monospace"}),
            html.Div(label, style={"color": TEXT_MUTED, "fontSize": "11px"}),
        ])

    pf_color = ACCENT_GREEN if pf > 1.5 else (ACCENT_RED if pf < 1 else TEXT_PRIMARY)
    pf_str   = f"{pf:.2f}" if pf != float("inf") else "∞"

    return [
        stat("Total Trades", str(total)),
        stat("Win Rate",  f"{win_pct:.1f}%",
             color=ACCENT_GREEN if win_pct > 50 else ACCENT_RED),
        stat("Profit Factor", pf_str, color=pf_color),
        stat("Avg P&L",   f"${avg_pnl:+.2f}",
             color=ACCENT_GREEN if avg_pnl > 0 else ACCENT_RED),
        stat("Total P&L", f"${sum(t['net_pnl'] for t in trades):+.2f}",
             color=ACCENT_GREEN if sum(t["net_pnl"] for t in trades) > 0 else ACCENT_RED),
    ]


def render_table(trades: list[dict]) -> list:
    if not trades:
        return _table_placeholder()

    rows = []
    cum  = 0.0
    wins = 0
    for i, t in enumerate(trades, 1):
        cum  += t["net_pnl"]
        wins += (t["net_pnl"] > 0)
        wr    = wins / i * 100
        rows.append({
            "#":          i,
            "Date":       t["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "Dir":        t["direction"],
            "Reason":     t["exit_reason"],
            "Entry":      f"{t['entry_price']:.4f}",
            "Exit":       f"{t['exit_price']:.4f}",
            "net_pnl":    round(t["net_pnl"], 2),
            "P&L ($)":    f"${t['net_pnl']:+.2f}",
            "Cum P&L":    f"${cum:+.2f}",
            "Win Rate":   f"{wr:.0f}%",
        })

    rows.reverse()   # newest first
    display_cols = ["#", "Date", "Dir", "Reason", "Entry", "Exit",
                    "P&L ($)", "Cum P&L", "Win Rate"]
    cols = [{"name": c, "id": c} for c in display_cols]

    style_data_cond = [
        {
            "if": {"filter_query": "{net_pnl} > 0"},
            "backgroundColor": "#1a2e1a",
        },
        {
            "if": {"filter_query": "{net_pnl} < 0"},
            "backgroundColor": "#2e1a1a",
        },
    ]

    return [dash_table.DataTable(
        data                   = rows,
        columns                = cols,
        style_header           = TABLE_STYLE_HEADER,
        style_cell             = {**TABLE_STYLE_CELL, "fontSize": "12px"},
        style_data_conditional = style_data_cond,
        page_size              = 20,
        sort_action            = "native",
        filter_action          = "native",
        style_table            = {"overflowX": "auto"},
    )]


def _empty_chart(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=12, color=TEXT_MUTED),
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
