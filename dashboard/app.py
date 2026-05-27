"""
Trading Dashboard — Dash application factory and entry point.

Launch:
    python -m dashboard.app          # from trading-agent/ directory
    ./dashboard/run_dashboard.sh     # convenience wrapper

Runs on http://localhost:8051 and opens the browser automatically.
"""
from __future__ import annotations

import threading
import time
import webbrowser

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.layout import theme
from dashboard.layout.header import build_header
from dashboard.layout.section_account import build_section as account_section
from dashboard.layout.section_portfolio import build_section as portfolio_section
from dashboard.layout.section_crypto import build_section as crypto_section
from dashboard.layout.section_strategy import build_section as strategy_section
from dashboard.layout.section_trades import build_section as trades_section
from dashboard.layout.section_divergence import build_section as divergence_section
from dashboard.callbacks.refresh import register_callbacks

PORT = 8051
HOST = "127.0.0.1"


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title="Trading Dashboard",
    )

    app.layout = html.Div(
        style={
            "backgroundColor": theme.BG_MAIN,
            "minHeight":       "100vh",
            "fontFamily":      "system-ui, -apple-system, sans-serif",
            "color":           theme.TEXT_PRIMARY,
        },
        children=[
            # Invisible interval timers
            dcc.Interval(id="interval-fast", interval=30_000,   n_intervals=0),
            dcc.Interval(id="interval-slow", interval=300_000,  n_intervals=0),

            # Header
            build_header(),

            # Page body
            html.Div(
                style={"padding": "20px 24px", "maxWidth": "1600px", "margin": "0 auto"},
                children=[
                    account_section(),
                    html.Div(style={"height": "4px"}),
                    portfolio_section(),
                    html.Div(style={"height": "4px"}),
                    crypto_section(),
                    html.Div(style={"height": "4px"}),
                    strategy_section(),
                    html.Div(style={"height": "4px"}),
                    trades_section(),
                    html.Div(style={"height": "4px"}),
                    divergence_section(),
                    html.Div(style={"height": "20px"}),
                ],
            ),
        ],
    )

    register_callbacks(app)
    return app


def _open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    import sys

    app = create_app()

    # Auto-open browser unless --no-browser flag is passed
    if "--no-browser" not in sys.argv:
        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Trading Dashboard                       │")
    print(f"  │  http://{HOST}:{PORT}                   │")
    print(f"  │  Ctrl+C to stop                          │")
    print(f"  └─────────────────────────────────────────┘\n")

    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
