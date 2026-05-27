"""Tool registry — exposes all tool specs and callable functions."""
from tools.backtest_tool import TOOL_SPEC as BACKTEST_SPEC, run_backtest
from tools.optimize_tool import TOOL_SPEC as OPTIMIZE_SPEC, optimize
from tools.walk_forward_tool import TOOL_SPEC as WALK_FORWARD_SPEC, walk_forward
from tools.deploy_tool import TOOL_SPEC as DEPLOY_SPEC, deploy_params
from tools.monitor_tool import TOOL_SPEC as MONITOR_SPEC, get_live_status
from tools.fundamental_optimize_tool import TOOL_SPEC as FUND_OPT_SPEC, optimize_fundamental
from tools.fundamental_backtest_tool import TOOL_SPEC as FUND_BT_SPEC, run_fundamental_backtest
from tools.screen_tool import TOOL_SPEC as SCREEN_SPEC, screen_fundamentals
from tools.universe_tool import TOOL_SPEC as UNIVERSE_SPEC, get_universe
from tools.walk_forward_fundamental_tool import TOOL_SPEC as FUND_WF_SPEC, run_fundamental_walk_forward
from tools.sentiment_optimize_tool import TOOL_SPEC as SENT_OPT_SPEC, optimize_sentiment
from tools.sentiment_backtest_tool import TOOL_SPEC as SENT_BT_SPEC, run_sentiment_backtest
from tools.sentiment_screen_tool import TOOL_SPEC as SENT_SCREEN_SPEC, screen_sentiment_tickers

ALL_TOOL_SPECS = [
    BACKTEST_SPEC,
    OPTIMIZE_SPEC,
    WALK_FORWARD_SPEC,
    DEPLOY_SPEC,
    MONITOR_SPEC,
    FUND_OPT_SPEC,
    FUND_BT_SPEC,
    SCREEN_SPEC,
    UNIVERSE_SPEC,
    FUND_WF_SPEC,
    SENT_OPT_SPEC,
    SENT_BT_SPEC,
    SENT_SCREEN_SPEC,
]

TOOL_DISPATCH = {
    "run_backtest":                  run_backtest,
    "run_optimization":              optimize,
    "run_walk_forward":              walk_forward,
    "deploy_params":                 deploy_params,
    "get_live_status":               get_live_status,
    "run_fundamental_optimization":  optimize_fundamental,
    "run_fundamental_backtest":      run_fundamental_backtest,
    "screen_fundamentals":           screen_fundamentals,
    "get_universe":                  get_universe,
    "run_fundamental_walk_forward":  run_fundamental_walk_forward,
    "run_sentiment_optimization":    optimize_sentiment,
    "run_sentiment_backtest":        run_sentiment_backtest,
    "screen_sentiment":              screen_sentiment_tickers,
}
