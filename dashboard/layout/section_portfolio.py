"""Portfolio performance: equity curve vs SPY benchmark."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from .theme import (
    CARD_STYLE, SECTION_HEADER, PLOTLY_LAYOUT,
    ACCENT_TEAL, ACCENT_YELLOW, TEXT_MUTED, BG_CARD, BORDER, ACCENT_BLUE,
)

_RANGE_BUTTONS = [
    dict(count=7,  label="1W", step="day",   stepmode="backward"),
    dict(count=1,  label="1M", step="month", stepmode="backward"),
    dict(count=3,  label="3M", step="month", stepmode="backward"),
    dict(             label="All", step="all"),
]


def build_section() -> html.Div:
    return html.Div([
        html.P("Portfolio Performance", style=SECTION_HEADER),
        html.Div(
            style=CARD_STYLE,
            children=[
                dcc.Graph(
                    id="portfolio-chart",
                    figure=_empty_figure("Waiting for trade history…"),
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "340px"},
                ),
            ],
        ),
    ])


def build_figure(
    equity_df: pd.DataFrame,
    spy_series: pd.Series,
    account_equity: float | None,
) -> go.Figure:
    """
    Build the dual-line equity vs SPY chart.

    equity_df   : output of state_reader.build_equity_series()
    spy_series  : output of benchmark.get_benchmark_series()
    """
    fig = go.Figure()
    layout = dict(PLOTLY_LAYOUT)

    has_trades = not equity_df.empty

    if has_trades:
        # Normalize strategy equity to 100 at first point
        base_equity = equity_df["equity"].iloc[0]
        norm_equity = (equity_df["equity"] / base_equity * 100).round(4)
        align_date  = equity_df["timestamp"].iloc[0]
        first_date_str = pd.Timestamp(align_date).strftime("%b %d, %Y")

        fig.add_trace(go.Scatter(
            x    = equity_df["timestamp"],
            y    = norm_equity,
            mode = "lines",
            name = "Bot Strategy",
            line = dict(color=ACCENT_TEAL, width=2),
            hovertemplate=(
                "<b>Strategy</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Index: %{y:.2f}<br>"
                f"Equity: $%{{customdata:,.2f}}<extra></extra>"
            ),
            customdata = equity_df["equity"],
        ))

        layout["title"] = dict(
            text=f"Portfolio vs SPY  ·  Since {first_date_str}",
            font=dict(size=13, color=TEXT_MUTED),
            x=0, pad=dict(l=4),
        )
    else:
        # No trades — show annotation
        fig.add_annotation(
            text="No trades recorded yet — SPY benchmark shown",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=TEXT_MUTED),
        )
        # Anchor SPY from 90 days ago
        from datetime import datetime, timedelta
        align_date = datetime.utcnow() - timedelta(days=90)
        layout["title"] = dict(
            text="Portfolio vs SPY  ·  (No trades yet)",
            font=dict(size=13, color=TEXT_MUTED),
            x=0, pad=dict(l=4),
        )

    if not spy_series.empty:
        fig.add_trace(go.Scatter(
            x    = spy_series.index,
            y    = spy_series.values,
            mode = "lines",
            name = "SPY",
            line = dict(color=ACCENT_YELLOW, width=1.5, dash="dot"),
            hovertemplate=(
                "<b>SPY</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Index: %{y:.2f}<extra></extra>"
            ),
        ))

    # Reference line at 100
    fig.add_hline(
        y=100, line_dash="dash",
        line_color="rgba(139,148,158,0.3)",
        line_width=1,
    )

    layout["yaxis"]["title"] = "Normalized (base=100)"
    layout["yaxis"]["tickformat"] = ".1f"
    layout["xaxis"]["rangeselector"] = dict(
        buttons=_RANGE_BUTTONS,
        bgcolor=BG_CARD,
        activecolor=ACCENT_BLUE,
        bordercolor=BORDER,
        font=dict(color=TEXT_MUTED, size=11),
        x=0, y=1.08,
    )
    layout["xaxis"]["rangeslider"] = dict(visible=False)

    fig.update_layout(**layout)
    return fig


def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color=TEXT_MUTED),
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    return fig
