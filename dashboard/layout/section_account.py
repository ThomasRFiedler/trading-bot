"""Account overview — 6 KPI cards."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from .theme import (
    CARD_STYLE, KPI_VALUE_STYLE, KPI_LABEL_STYLE, KPI_SUBVALUE_STYLE,
    SECTION_HEADER, ACCENT_GREEN, ACCENT_RED, TEXT_MUTED, BG_CARD, BORDER,
)

PLACEHOLDER = "—"


def build_section() -> html.Div:
    """Static shell — values populated by callback."""
    return html.Div([
        html.P("Account Overview", style=SECTION_HEADER),
        dbc.Row([
            dbc.Col(_kpi_card("account-value",    "Account Value"),   xs=6, md=4, lg=2),
            dbc.Col(_kpi_card("market-value",     "Market Value"),    xs=6, md=4, lg=2),
            dbc.Col(_kpi_card("total-gain-loss",  "Total Gain/Loss"), xs=6, md=4, lg=2),
            dbc.Col(_kpi_card("today-gain-loss",  "Today's Gain"),    xs=6, md=4, lg=2),
            dbc.Col(_kpi_card("buying-power",     "Buying Power"),    xs=6, md=4, lg=2),
            dbc.Col(_kpi_card("positions",        "Open Positions"),  xs=6, md=4, lg=2),
        ], className="g-2"),
    ])


def _kpi_card(id_prefix: str, label: str) -> html.Div:
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.Div(PLACEHOLDER, id=f"{id_prefix}-value",
                     style=KPI_VALUE_STYLE),
            html.Div(PLACEHOLDER, id=f"{id_prefix}-sub",
                     style=KPI_SUBVALUE_STYLE),
            html.Div(label, style=KPI_LABEL_STYLE),
        ],
    )


# ── Render helpers (called from callbacks) ────────────────────────────────────

def _color(value: float) -> str:
    if value > 0:
        return ACCENT_GREEN
    if value < 0:
        return ACCENT_RED
    return TEXT_MUTED


def render_account_value(account_equity: float | None, cum_pnl: float) -> tuple:
    base   = account_equity if account_equity else 10_000.0
    total  = base + cum_pnl
    value  = f"${total:,.2f}"
    sub    = ""
    return value, sub


def render_market_value(in_position: bool, entry_price: float | None,
                        position_size: float = 100.0) -> tuple:
    if in_position and entry_price:
        mv = entry_price * (position_size / entry_price) if entry_price else position_size
        # Approximate: position_size is dollar-based in trading-app
        value = f"${position_size:,.2f}"
        sub   = f"@ {entry_price:.4f}"
    else:
        value, sub = "$0.00", "No position"
    return value, sub


def render_total_gain(cum_pnl: float, account_equity: float | None) -> tuple:
    base  = account_equity if account_equity else 10_000.0
    pct   = cum_pnl / base * 100 if base else 0
    style = {**KPI_VALUE_STYLE, "color": _color(cum_pnl)}
    value = f"${cum_pnl:+,.2f}"
    sub   = f"{pct:+.2f}%"
    return value, sub, style


def render_today_gain(today_pnl: float) -> tuple:
    style = {**KPI_VALUE_STYLE, "color": _color(today_pnl)}
    value = f"${today_pnl:+,.2f}"
    sub   = ""
    return value, sub, style


def render_buying_power(account_equity: float | None, cum_pnl: float,
                        in_position: bool, position_size: float = 100.0) -> tuple:
    base  = account_equity if account_equity else 10_000.0
    total = base + cum_pnl
    mv    = position_size if in_position else 0.0
    bp    = total - mv
    return f"${bp:,.2f}", ""


def render_positions(in_position: bool, position_type: str | None,
                     ticker: str = "AAPL") -> tuple:
    if in_position:
        direction = (position_type or "").upper()
        return "1", f"{direction} {ticker}"
    return "0", "Flat"
