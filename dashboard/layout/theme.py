"""Dark theme constants and reusable style dicts."""

# ── Palette ───────────────────────────────────────────────────────────────────
BG_MAIN       = "#0d1117"
BG_CARD       = "#161b22"
BG_CARD_ALT   = "#1c2128"
BG_HEADER     = "#21262d"
ACCENT_GREEN  = "#3fb950"
ACCENT_RED    = "#f85149"
ACCENT_BLUE   = "#58a6ff"
ACCENT_TEAL   = "#39d0d8"
ACCENT_YELLOW = "#e3b341"
TEXT_PRIMARY  = "#e6edf3"
TEXT_MUTED    = "#8b949e"
TEXT_DIM      = "#484f58"
BORDER        = "#30363d"

PLOTLY_TEMPLATE = "plotly_dark"

# ── Reusable component styles ─────────────────────────────────────────────────
CARD_STYLE = {
    "backgroundColor": BG_CARD,
    "border":          f"1px solid {BORDER}",
    "borderRadius":    "8px",
    "padding":         "16px 20px",
    "marginBottom":    "12px",
}

SECTION_HEADER = {
    "color":        TEXT_MUTED,
    "fontSize":     "11px",
    "fontWeight":   "600",
    "letterSpacing":"0.08em",
    "textTransform":"uppercase",
    "marginBottom": "10px",
}

KPI_VALUE_STYLE = {
    "color":      TEXT_PRIMARY,
    "fontSize":   "24px",
    "fontWeight": "600",
    "lineHeight": "1.2",
    "marginBottom": "2px",
}

KPI_LABEL_STYLE = {
    "color":    TEXT_MUTED,
    "fontSize": "12px",
}

KPI_SUBVALUE_STYLE = {
    "fontSize": "13px",
    "marginTop": "2px",
}

TABLE_STYLE_HEADER = {
    "backgroundColor": BG_HEADER,
    "color":           TEXT_MUTED,
    "fontWeight":      "600",
    "fontSize":        "11px",
    "border":          f"1px solid {BORDER}",
    "textAlign":       "left",
    "padding":         "8px 12px",
}

TABLE_STYLE_CELL = {
    "backgroundColor": BG_CARD,
    "color":           TEXT_PRIMARY,
    "fontSize":        "13px",
    "border":          f"1px solid {BORDER}",
    "padding":         "7px 12px",
    "fontFamily":      "monospace",
}

TABLE_STYLE_DATA_CONDITIONAL = [
    {
        "if": {"filter_query": "{net_pnl} > 0"},
        "backgroundColor": "#1a2e1a",
        "color":           ACCENT_GREEN,
    },
    {
        "if": {"filter_query": "{net_pnl} < 0"},
        "backgroundColor": "#2e1a1a",
        "color":           ACCENT_RED,
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": BG_CARD_ALT,
        "border":          f"1px solid {ACCENT_BLUE}",
    },
]

# ── Plotly figure base layout ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template        = PLOTLY_TEMPLATE,
    paper_bgcolor   = BG_CARD,
    plot_bgcolor    = BG_CARD,
    font            = dict(color=TEXT_PRIMARY, family="system-ui, sans-serif", size=12),
    xaxis           = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                           linecolor=BORDER, tickcolor=TEXT_DIM),
    yaxis           = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                           linecolor=BORDER, tickcolor=TEXT_DIM),
    legend          = dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    margin          = dict(l=10, r=10, t=36, b=10),
    hovermode       = "x unified",
)
