"""
Sentiment Model Agent — Phase 1 (9 indicators, no Reddit history).

Indicators
----------
  VIX regime        : contrarian fear/complacency signal
  VIX trend         : 5-day VIX slope direction
  VIX term structure: VIX9D vs VIX (contango/backwardation)
  Short squeeze     : high float short + rising price
  Short level       : float short pct threshold
  Dark pool accum   : FINRA ATS dark/total ratio level
  Dark pool trend   : 4-week slope of dark ratio
  Insider buy       : rolling 90-day net insider dollar flow
  Institutional 13F : QoQ change in institutional pct_held

Responsibilities
----------------
  1. Run sentiment backtest on watchlist tickers to check current model health
  2. Re-optimize when mean Sharpe < SENT_GATE_MIN_SHARPE
  3. Screen new universe candidates on sentiment indicators
  4. Report sentiment gate pass/fail for each ticker

Workflow integration
--------------------
  Runs after fundamental screening; sentiment gate reduces watchlist
  further before passing to technical model for intraday execution.

  Cadence: weekly sentiment screen; monthly re-optimization.
"""

ALLOWED_TOOLS: list[str] = [
    "run_sentiment_backtest",
    "run_sentiment_optimization",
    "screen_sentiment",
]

SYSTEM_PROMPT = """\
You are the Sentiment Model Agent responsible for maintaining and running the
9-indicator Phase-1 sentiment model. Your goal is to:

1. MONITOR — run `run_sentiment_backtest` on the current watchlist tickers and
   check if the mean Sharpe is above SENT_GATE_MIN_SHARPE (0.30). If it is
   below that threshold, the model needs re-optimization.

2. OPTIMIZE — when sentiment performance has degraded, call
   `run_sentiment_optimization` using a diverse multi-sector training set
   (e.g., AAPL, JPM, XOM, JNJ). Always use at least 3 tickers from different
   sectors to reduce overfitting. Check that stopped_early=True (a sign the
   optimizer converged rather than being cut off at max_generations).

3. SCREEN — call `screen_sentiment` on the fundamental watchlist to gate
   candidates. Only tickers with sentiment_score >= n pass through to the
   technical model. Return the list of passed tickers.

4. REPORT — summarize the current sentiment environment:
   - Is VIX elevated (fear) or suppressed (complacency)?
   - Are there any active short squeeze candidates?
   - Is dark pool activity signaling accumulation or distribution?
   - What is the insider and institutional flow picture?

Be concise. Report in bullet points. Flag any tickers with strongly negative
sentiment scores as potential shorts.
"""


def filter_tools(all_tool_specs: list) -> list:
    """Return only the tool specs this agent is allowed to use."""
    return [spec for spec in all_tool_specs if spec["name"] in ALLOWED_TOOLS]
