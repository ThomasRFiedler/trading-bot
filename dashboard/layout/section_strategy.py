"""Bot strategy: params card, indicator weights bar chart, deployment history."""
from __future__ import annotations

import plotly.graph_objects as go
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc

from .theme import (
    CARD_STYLE, SECTION_HEADER, KPI_LABEL_STYLE, PLOTLY_LAYOUT,
    ACCENT_TEAL, ACCENT_RED, TEXT_PRIMARY, TEXT_MUTED, TEXT_DIM,
    BG_CARD, BG_HEADER, BORDER, TABLE_STYLE_HEADER, TABLE_STYLE_CELL,
)


def build_section() -> html.Div:
    return html.Div([
        html.P("Bot Strategy", style=SECTION_HEADER),
        dbc.Row([
            # Left column: params card + weights chart
            dbc.Col([
                html.Div(id="strategy-params-card",
                         style=CARD_STYLE,
                         children=_params_placeholder()),
                html.Div(style=CARD_STYLE, children=[
                    html.P("Indicator Weights", style={**KPI_LABEL_STYLE, "marginBottom": "8px"}),
                    dcc.Graph(
                        id="weights-chart",
                        figure=_empty_weights(),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "320px"},
                    ),
                ]),
            ], md=5),

            # Right column: deployment history table
            dbc.Col([
                html.Div(style={**CARD_STYLE, "height": "100%"}, children=[
                    html.P("Deployment History", style={**KPI_LABEL_STYLE, "marginBottom": "8px"}),
                    html.Div(id="model-history-table", children=_history_placeholder()),
                ]),
            ], md=7),
        ], className="g-2"),
    ])


def _params_placeholder() -> list:
    return [html.P("No params deployed", style={"color": TEXT_MUTED, "fontSize": "13px"})]


def _history_placeholder() -> list:
    return [html.P("No deployment history", style={"color": TEXT_MUTED, "fontSize": "13px"})]


# ── Render helpers ────────────────────────────────────────────────────────────

def render_params_card(params: dict, model_type: str, deployed_at: str,
                       ticker: str) -> list:
    if not params:
        return _params_placeholder()

    n           = params.get("n", 0)
    sl          = params.get("stop_loss", 0)
    tp          = params.get("take_profit", 0)
    sharpe      = params.get("sharpe", 0)
    model_label = model_type.capitalize()

    def row(label, value, color=TEXT_PRIMARY):
        return html.Div(style={"display": "flex", "justifyContent": "space-between",
                               "marginBottom": "6px"}, children=[
            html.Span(label, style={"color": TEXT_MUTED, "fontSize": "12px"}),
            html.Span(value, style={"color": color, "fontSize": "13px",
                                    "fontWeight": "600", "fontFamily": "monospace"}),
        ])

    return [
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "marginBottom": "10px"}, children=[
            html.Span(f"{model_label} Model — {ticker}",
                      style={"color": TEXT_PRIMARY, "fontSize": "13px", "fontWeight": "600"}),
            html.Span("DEPLOYED", style={
                "color": ACCENT_TEAL, "fontSize": "10px", "fontWeight": "700",
                "backgroundColor": "#1a2e2e", "padding": "2px 7px",
                "borderRadius": "4px", "letterSpacing": "0.06em",
            }),
        ]),
        row("Deployed at",  deployed_at[:16] if deployed_at else "—"),
        row("n (threshold)", f"{n:.4f}"),
        row("Stop Loss",    f"{sl:.1%}"),
        row("Take Profit",  f"{tp:.1%}"),
        row("Train Sharpe", f"{sharpe:.4f}",
            color=ACCENT_TEAL if sharpe > 1 else TEXT_PRIMARY),
    ]


def render_weights_chart(weight_pairs: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bar chart of indicator weights, sorted descending."""
    if not weight_pairs:
        return _empty_weights()

    names   = [p[0] for p in weight_pairs]
    weights = [p[1] for p in weight_pairs]

    # Color: teal gradient for active, muted red for zero
    max_w = max(weights) if max(weights) > 0 else 1.0
    colors = []
    for w in weights:
        if w == 0:
            colors.append("#3a1a1a")   # zeroed out → dark red
        else:
            alpha = 0.4 + 0.6 * (w / max_w)
            r  = int(57  * alpha)
            g  = int(208 * alpha)
            b  = int(216 * alpha)
            colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure(go.Bar(
        x           = weights,
        y           = names,
        orientation = "h",
        marker_color= colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        text        = [f"{w:.3f}" if w > 0 else "off" for w in weights],
        textposition= "outside",
        textfont    = dict(size=10, color=TEXT_MUTED),
    ))

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        xaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=True,
                   zerolinecolor=BORDER, title=None, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed",
                   tickfont=dict(size=11)),
        margin=dict(l=0, r=60, t=10, b=10),
        bargap=0.25,
        hovermode="y unified",
    )
    fig.update_layout(**layout)
    return fig


def render_history_table(df) -> list:
    """Render deployment history as a Dash DataTable."""
    import pandas as pd
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return _history_placeholder()

    display_cols = ["deployed_at", "model_type", "ticker",
                    "sharpe", "max_drawdown", "profit_factor",
                    "total_trades", "overfit_gap"]
    col_labels = {
        "deployed_at":   "Deployed",
        "model_type":    "Model",
        "ticker":        "Ticker",
        "sharpe":        "Sharpe",
        "max_drawdown":  "Max DD",
        "profit_factor": "PF",
        "total_trades":  "Trades",
        "overfit_gap":   "Overfit",
    }

    cols = [{"name": col_labels.get(c, c), "id": c}
            for c in display_cols if c in df.columns]

    style_data_cond = [
        # Highlight first row (most recent)
        {
            "if": {"row_index": 0},
            "backgroundColor": "#1a2e1a",
            "border": f"1px solid {ACCENT_TEAL}",
        },
        # Red overfit gap
        {
            "if": {"filter_query": "{overfit_gap} > 1.5", "column_id": "overfit_gap"},
            "color": ACCENT_RED,
        },
    ]

    return [dash_table.DataTable(
        data            = df[display_cols].head(10).to_dict("records"),
        columns         = cols,
        style_header    = TABLE_STYLE_HEADER,
        style_cell      = {**TABLE_STYLE_CELL, "minWidth": "60px",
                           "fontFamily": "monospace", "fontSize": "11px"},
        style_data_conditional = style_data_cond,
        page_size       = 10,
        sort_action     = "native",
        style_table     = {"overflowX": "auto"},
    )]


def _empty_weights() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No params loaded", xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=12, color=TEXT_MUTED),
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
