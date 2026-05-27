"""Top status bar: connection pill, refresh timestamp, title."""
import dash_bootstrap_components as dbc
from dash import html

from .theme import (
    BG_HEADER, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW,
    TEXT_PRIMARY, TEXT_MUTED, BORDER,
)


def build_header() -> html.Div:
    return html.Div(
        style={
            "backgroundColor": BG_HEADER,
            "borderBottom":    f"1px solid {BORDER}",
            "padding":         "12px 24px",
            "display":         "flex",
            "alignItems":      "center",
            "justifyContent":  "space-between",
        },
        children=[
            # Left: title
            html.Div([
                html.Span("⬡ ", style={"color": ACCENT_TEAL, "fontSize": "18px"}),
                html.Span("Trading Dashboard", style={
                    "color":      TEXT_PRIMARY,
                    "fontSize":   "15px",
                    "fontWeight": "600",
                }),
            ]),

            # Centre: connection pill + ticker
            html.Div(
                id="header-status",
                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                children=_status_placeholder(),
            ),

            # Right: last-refresh time
            html.Div(
                id="header-refresh",
                style={"color": TEXT_MUTED, "fontSize": "12px"},
                children="—",
            ),
        ],
    )


def _status_placeholder():
    return [
        html.Span("● OFFLINE", style={
            "color":        ACCENT_RED,
            "fontSize":     "12px",
            "fontWeight":   "600",
            "letterSpacing":"0.05em",
        }),
    ]


def render_status(connected: bool, stale: bool, ticker: str = "AAPL") -> list:
    if connected and not stale:
        color, label = ACCENT_GREEN, "● LIVE"
    elif connected and stale:
        color, label = ACCENT_YELLOW, "● STALE"
    else:
        color, label = ACCENT_RED, "● OFFLINE"

    return [
        html.Span(label, style={
            "color":        color,
            "fontSize":     "12px",
            "fontWeight":   "600",
            "letterSpacing":"0.05em",
        }),
        html.Span(ticker, style={
            "color":        TEXT_MUTED,
            "fontSize":     "12px",
            "backgroundColor": "#30363d",
            "padding":      "2px 8px",
            "borderRadius": "4px",
        }),
    ]


# Need this import here to avoid circular — only used in render_status label
ACCENT_TEAL = "#39d0d8"
