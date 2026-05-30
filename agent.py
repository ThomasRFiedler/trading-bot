"""
Trading Agent — Entry Point

Workflow overview
-----------------
  Technical loop  (every 30 min, market hours)
    → monitor live metrics → if Sharpe degraded, re-optimize → deploy if gates pass

  Fundamental loop  (weekly)
    → get_universe (low market cap) → screen_fundamentals → update watchlist
    → if monthly cadence, re-optimize fundamental model on watchlist tickers

  Full scheduler  (--workflow)
    → runs both loops on their respective cadences; the fundamental watchlist
      determines which tickers the technical model is allowed to trade

Usage:
    python agent.py                          # Run technical agent once
    python agent.py --agent fundamental      # Run fundamental agent once
    python agent.py --screen                 # Universe → screen → save watchlist
    python agent.py --loop                   # Technical loop, market hours
    python agent.py --workflow               # Full stock scheduler: technical + fundamental
    python agent.py --workflow crypto        # Crypto scheduler: 24/7 technical optimize loop
    python agent.py --workflow all           # Both stock + crypto schedulers in parallel
    python agent.py --check                  # Dry-run: validate config + paths
    python agent.py --backtest-only          # Backtest deployed params, no deploy
    python agent.py --optimize-and-deploy    # Full technical optimize → deploy cycle

OpenShell (sandboxed):
    openshell sandbox create -- python agent.py --check
    openshell sandbox create -- python agent.py --workflow
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app.agent")

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_market_hours() -> bool:
    """True during NYSE market hours (9:30–16:00 ET Mon–Fri)."""
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:
        return False
    now_et   = (now_utc.hour - 4) % 24
    now_min  = now_utc.minute
    total    = now_et * 60 + now_min
    return 9 * 60 + 30 <= total <= 16 * 60


def _days_since(timestamp_str: str | None) -> float:
    """Return days elapsed since an ISO timestamp string. inf if None."""
    if not timestamp_str:
        return float("inf")
    try:
        ts = datetime.fromisoformat(timestamp_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    except Exception:
        return float("inf")


def _load_schedule() -> dict:
    """Load last-run timestamps from registry/schedule.json."""
    path = config.REGISTRY_DIR / "schedule.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_schedule(schedule: dict) -> None:
    config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REGISTRY_DIR / "schedule.json", "w") as f:
        json.dump(schedule, f, indent=2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Individual actions
# ---------------------------------------------------------------------------

def _check() -> bool:
    """Validate configuration, paths, and API key."""
    logger.info("=== Configuration check ===")
    ok = True

    for name, path in [
        ("stock-signal-testing", config.SIGNAL_TESTING_DIR),
        ("trading-app",          config.TRADING_APP_DIR),
        ("registry",             config.REGISTRY_DIR),
    ]:
        exists = path.exists()
        logger.info(f"  {name:30s} {path}  [{'OK' if exists else 'MISSING'}]")
        if not exists:
            ok = False

    has_key = bool(config.ANTHROPIC_API_KEY)
    logger.info(f"  {'ANTHROPIC_API_KEY':30s} {'set' if has_key else 'NOT SET'}")
    if not has_key:
        ok = False

    deployed  = config.load_deployed_params()
    fund_p    = config.load_fundamental_params()
    watchlist = config.WATCHLIST_FILE.exists()
    logger.info(f"  {'Technical params':30s} {'found' if deployed else 'none'}")
    logger.info(f"  {'Fundamental params':30s} {'found' if fund_p else 'none'}")
    logger.info(f"  {'Watchlist':30s} {'found' if watchlist else 'none (will use default)'}")
    logger.info(f"  {'Universe tier':30s} {config.UNIVERSE_TIER}")
    logger.info(f"  {'Screen every':30s} {config.SCREEN_INTERVAL_DAYS} days")
    logger.info(f"  {'Re-opt fundamental every':30s} {config.REOPT_INTERVAL_DAYS} days")

    logger.info("=== Check complete — %s ===", "PASS" if ok else "FAIL")
    return ok


def run_screen(save: bool = True) -> list[str]:
    """
    Run universe → fundamental screen → save watchlist.
    Returns the list of tickers that passed the screen.
    """
    import sys
    sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
    from fundamental_testing.run_screener import run_screener

    fund_params = config.load_fundamental_params()
    params_path = str(config.FUNDAMENTAL_PARAMS_FILE) if fund_params else None

    logger.info(f"Running universe screen  (tier={config.UNIVERSE_TIER})...")
    results = run_screener(
        tier=config.UNIVERSE_TIER,
        max_universe=config.UNIVERSE_MAX,
        top_n=10,
        params_path=params_path,
        lookback_years=1.0,
        save_path=str(config.WATCHLIST_FILE) if save else None,
        verbose=True,
    )

    passed = [r["ticker"] for r in results if r.get("passed_screen")]
    logger.info(f"Screen complete — {len(passed)}/{len(results)} passed: {passed}")
    return passed


def _load_watchlist() -> list[str]:
    """Load active watchlist from registry; fall back to config.WATCHLIST."""
    if config.WATCHLIST_FILE.exists():
        with open(config.WATCHLIST_FILE) as f:
            data = json.load(f)
        tickers = data.get("tickers") or []
        if tickers:
            return tickers
    return config.WATCHLIST


def run_once(agent_name: str = "technical") -> None:
    from agents.orchestrator import run_agent
    logger.info(f"Running {agent_name} agent...")
    summary = run_agent(agent_name)
    logger.info(f"{agent_name} agent result:\n{summary}")


def run_loop(agent_name: str = "technical") -> None:
    interval_sec = config.CHECK_INTERVAL_MIN * 60
    logger.info(f"Starting {agent_name} loop — every {config.CHECK_INTERVAL_MIN}m during market hours.")
    while True:
        if _is_market_hours():
            run_once(agent_name)
        else:
            logger.info("Market closed — waiting...")
        time.sleep(interval_sec)


def backtest_only() -> None:
    from tools.backtest_tool import run_backtest
    from tools.monitor_tool import get_live_status

    status = get_live_status()
    params = status.get("deployed_params")
    if not params:
        logger.error("No deployed params found. Run --optimize-and-deploy first.")
        sys.exit(1)

    logger.info(f"Backtesting deployed params for {config.TICKER}...")
    metrics = run_backtest(ticker=config.TICKER, params=params)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")


def _fetch_stock_data(ticker: str, interval: str) -> dict:
    """
    Fetch intraday price data for a stock ticker.

    Tries IBKR Gateway first (6 months of history when USE_IBKR_DATA=true),
    falls back to Yahoo Finance (60-day cap) if IBKR is unavailable.
    """
    if config.USE_IBKR_DATA:
        try:
            from signal_testing.data_ibkr import fetch_data_ibkr
            logger.info(
                f"Fetching {config.IBKR_MONTHS}m of {ticker} @ {interval} from IBKR..."
            )
            data = fetch_data_ibkr(
                ticker, interval=interval, n_months=config.IBKR_MONTHS,
                host=config.IB_HOST, port=config.IB_PORT,
                client_id=config.IB_DATA_CLIENT_ID,
            )
            bars = len(data["price"])
            logger.info(
                f"IBKR data: {bars} bars  "
                f"({data['price'].index[0].date()} → {data['price'].index[-1].date()})"
            )
            return data
        except Exception as exc:
            logger.warning(f"IBKR fetch failed ({exc}) — falling back to Yahoo Finance")

    from signal_testing.data import fetch_data
    return fetch_data(ticker, config.TIME_FRAME, interval)


def optimize_and_deploy(crypto: bool = False) -> None:
    from tools.optimize_tool import optimize
    from tools.backtest_tool import run_backtest
    from tools.walk_forward_tool import walk_forward
    from tools.deploy_tool import deploy_params

    ticker   = config.CRYPTO_TICKER if crypto else config.TICKER
    interval = config.INTERVAL
    label    = f"{ticker} [crypto]" if crypto else ticker

    # Fetch data once — reused by optimizer, backtest, and walk-forward.
    # For stocks, tries IBKR (6 months) then falls back to Yahoo (60d).
    if not crypto:
        preloaded_data = _fetch_stock_data(ticker, interval)
    else:
        preloaded_data = None   # crypto workflow fetches its own data

    logger.info(f"Optimizing {label}...")
    opt_result = optimize(
        ticker=ticker, crypto=crypto,
        preloaded_data=preloaded_data if not crypto else None,
    )
    logger.info(f"Train Sharpe: {opt_result['sharpe']:.3f}")

    metrics = run_backtest(
        ticker=ticker, params=opt_result, crypto=crypto,
        preloaded_data=preloaded_data if not crypto else None,
    )
    metrics["overfit_gap"] = opt_result["sharpe"] - metrics["sharpe_ratio"]

    # Walk-forward validation — required before deployment
    logger.info("Running walk-forward validation...")
    wf_result = walk_forward(
        ticker=ticker,
        preloaded_data=preloaded_data if not crypto else None,
    )
    if wf_result.get("error"):
        logger.warning(f"Walk-forward failed: {wf_result['error']} — proceeding without WF gate")
    else:
        metrics["mean_wf_test_sharpe"] = wf_result["mean_test_sharpe"]
        logger.info(
            f"Walk-forward: mean_test_sharpe={wf_result['mean_test_sharpe']:.3f}  "
            f"mean_overfit_gap={wf_result['mean_overfit_gap']:.3f}  "
            f"windows={len(wf_result['windows'])}  "
            f"bars={wf_result.get('bars_available', '?')}"
        )

    result = deploy_params(
        params=opt_result,
        metrics=metrics,
        model_type="technical",
        ticker=ticker,
        notes=f"Manual optimize-and-deploy from agent.py{'  [crypto]' if crypto else ''}",
        crypto=crypto,
        walk_forward_results=wf_result if not wf_result.get("error") else None,
    )

    if result["deployed"]:
        logger.info(
            f"Deployed — Sharpe={metrics['sharpe_ratio']:.3f}  "
            f"Trades={metrics['total_trades']}  DD={metrics['max_drawdown']:.1%}"
        )
        if result.get("review"):
            rv = result["review"]
            logger.info(
                f"Adversarial review: {rv.outcome}  "
                f"confidence={rv.confidence or 0.0:.2f}"
            )
    else:
        logger.warning(f"Deployment rejected — {result['failures']}")


# ---------------------------------------------------------------------------
# Full workflow scheduler
# ---------------------------------------------------------------------------

def _log_trader_state() -> None:
    """Read state.json and log current position and recent trade activity."""
    try:
        from dashboard.data.state_reader import read_state, read_trade_history, pnl_today
        state  = read_state()
        trades = read_trade_history()

        if state["stale"] or not state["connected"]:
            logger.info("[TRADER] Live trader not running / state file stale")
            return

        acct  = state["account_type"].upper()   # "PAPER" | "LIVE" | "UNKNOWN"
        mode  = state["trading_mode"] or "?"
        label = f"[{acct}/{mode}]"
        logger.info(f"[TRADER] {label} account connected")

        if state["in_position"]:
            logger.info(
                f"[TRADER] {label} OPEN {(state['position_type'] or '').upper()} "
                f"@ {state['entry_price']}  since {state['entry_time']}  "
                f"qty={state['quantity']}"
            )
        else:
            logger.info(f"[TRADER] {label} FLAT — no open position")

        recent = trades[-5:] if trades else []
        for t in recent:
            sign = "+" if t["net_pnl"] >= 0 else ""
            logger.info(
                f"[TRADER] {label}   {t['timestamp'].strftime('%m-%d %H:%M')} "
                f"{t['direction']:5s} [{t['exit_reason']:12s}] "
                f"P/L {sign}${t['net_pnl']:.2f}"
            )

        today_pnl = pnl_today(trades)
        sign = "+" if today_pnl >= 0 else ""
        logger.info(
            f"[TRADER] {label} Today: {sign}${today_pnl:.2f}  |  "
            f"Total trades: {state['total_trades']}  |  "
            f"Cumulative P/L: ${state['cumulative_pnl']:.2f}"
        )
    except Exception as exc:
        logger.debug(f"[TRADER] Could not read trader state: {exc}")


def run_workflow_crypto() -> None:
    """
    Crypto workflow scheduler — runs 24/7, technical optimize loop only.

    No market-hours gating (crypto trades around the clock).
    No fundamental/sentiment screen (equity indicators are not applicable).

    Overfitting guard: data is fetched once per cycle and split 80/20 into a
    training window and a held-out test window.  The optimizer never sees the
    test slice; the deploy gate uses the out-of-sample test Sharpe.

    Cadences
    --------
    Every CHECK_INTERVAL_MIN : Optimize on train slice → backtest on OOS test
                               slice → deploy if gates pass → log trader state

    State tracked in registry/schedule.json (key: last_crypto_check).
    """
    import sys as _sys
    _sys.path.insert(0, str(config.SIGNAL_TESTING_DIR))
    from signal_testing.data import fetch_data
    from signal_testing.data_ibkr import fetch_data_ibkr
    from signal_testing.walk_forward import _slice_data
    from tools.optimize_tool import optimize
    from tools.backtest_tool import run_backtest
    from tools.deploy_tool import deploy_params

    ticker        = config.CRYPTO_TICKER
    train_tickers = config.CRYPTO_TRAIN_TICKERS   # e.g. ["BTC", "ETH"]
    interval      = config.INTERVAL

    logger.info("=" * 60)
    logger.info("  CRYPTO WORKFLOW SCHEDULER  (24/7)")
    logger.info("=" * 60)
    logger.info(f"  Primary ticker  : {ticker}")
    logger.info(f"  Train tickers   : {train_tickers}")
    logger.info(f"  Interval        : {interval}")
    logger.info(f"  Check cadence   : every {config.CRYPTO_REOPT_INTERVAL_MIN} min")
    logger.info(f"  Train/test split: 80 / 20  (OOS test prevents overfitting)")
    logger.info("=" * 60)

    # Tickers whose IBKR data subscription is missing — skip IBKR for the
    # rest of this session rather than reconnecting and failing every cycle.
    _ibkr_no_permissions: set[str] = set()

    def _fetch_crypto(t: str) -> dict | None:
        """Try IBKR first, fall back to yahooquery. Returns None on total failure."""
        if t not in _ibkr_no_permissions:
            try:
                return fetch_data_ibkr(
                    ticker=t, interval=interval, crypto=True,
                    host=config.IB_HOST, port=config.IB_PORT,
                    client_id=config.IB_DATA_CLIENT_ID,
                )
            except PermissionError as exc:
                logger.warning(
                    f"[CRYPTO] IBKR market data not subscribed for {t} — "
                    f"switching to yahooquery for this session. "
                    f"To enable IBKR data: TWS → Market Data → Crypto by Kaiko."
                )
                _ibkr_no_permissions.add(t)
            except Exception as exc:
                logger.warning(f"[CRYPTO] IBKR fetch for {t} failed ({exc}), trying yahooquery...")

        try:
            return fetch_data(t, "60d", interval, crypto=True)
        except Exception as exc2:
            logger.error(f"[CRYPTO] All data sources failed for {t}: {exc2}")
            return None

    while True:
        schedule = _load_schedule()
        logger.info(f"[CRYPTO] ── Cycle start ──────────────────────────────")

        # ── Live trader status ────────────────────────────────────────────────
        _log_trader_state()

        # ── Fetch data for every train ticker, split each 80/20 ──────────────
        train_data_map = {}   # {ticker: train_slice}
        test_data      = None  # OOS slice for primary ticker only
        fetch_ok       = True

        for t in train_tickers:
            logger.info(f"[CRYPTO] Fetching {t} @ {interval}...")
            full = _fetch_crypto(t)
            if full is None:
                fetch_ok = False
                break
            n = len(full["price"])
            if n < 200:
                logger.error(f"[CRYPTO] {t}: only {n} bars — skipping cycle")
                fetch_ok = False
                break
            s = int(n * 0.8)
            train_data_map[t] = _slice_data(full, 0, s)
            if t == ticker:
                test_data = _slice_data(full, s, n)
                train_start = full["price"].index[0].date()
                split_date  = full["price"].index[s].date()
                test_end    = full["price"].index[-1].date()
                logger.info(
                    f"[CRYPTO] {t}: {n} bars  |  "
                    f"Train: {train_start} → {split_date} ({s} bars)  |  "
                    f"OOS: {split_date} → {test_end} ({n - s} bars)"
                )
            else:
                n_train = s
                logger.info(f"[CRYPTO] {t}: {n} bars  |  Train: {n_train} bars (co-training ticker)")

        if not fetch_ok or test_data is None:
            time.sleep(config.CRYPTO_REOPT_INTERVAL_MIN * 60 or 60)
            continue

        try:
            # ── Optimize simultaneously on all train ticker slices ────────────
            logger.info(f"[CRYPTO] Optimizing on {list(train_data_map)} training slices...")
            opt_result = optimize(
                ticker=ticker, crypto=True,
                preloaded_data_map=train_data_map,
            )
            logger.info(
                f"[CRYPTO] Train Sharpe: {opt_result['sharpe']:.3f}  |  "
                f"n={opt_result['n']:.3f}  "
                f"SL={opt_result['stop_loss']:.1%}  "
                f"TP={opt_result['take_profit']:.1%}  "
                f"gens={opt_result['generations_run']}"
            )

            # ── Out-of-sample backtest on held-out test slice ─────────────────
            metrics = run_backtest(
                ticker=ticker, params=opt_result,
                preloaded_data=test_data, crypto=True
            )
            metrics["overfit_gap"] = opt_result["sharpe"] - metrics["sharpe_ratio"]
            logger.info(
                f"[CRYPTO] OOS test  — Sharpe: {metrics['sharpe_ratio']:.3f}  |  "
                f"Trades: {metrics['total_trades']}  |  "
                f"DD: {metrics['max_drawdown']:.1%}  |  "
                f"Overfit gap: {metrics['overfit_gap']:.3f}"
            )

            # ── Deploy (gates use OOS test metrics) ───────────────────────────
            result = deploy_params(
                params=opt_result,
                metrics=metrics,
                model_type="technical",
                ticker=ticker,
                notes="Crypto workflow scheduler (OOS validated)",
                crypto=True,
                min_trades_override=config.OOS_GATE_MIN_TRADES,
            )

            if result["deployed"]:
                logger.info(
                    f"[CRYPTO] Deployed params → crypto_latest.json  "
                    f"(OOS Sharpe={metrics['sharpe_ratio']:.3f}  "
                    f"DD={metrics['max_drawdown']:.1%})"
                )
            else:
                logger.warning(f"[CRYPTO] Deployment rejected — {result['failures']}")

            schedule["last_crypto_check"] = _now_iso()
            _save_schedule(schedule)

        except Exception as exc:
            logger.error(f"[CRYPTO] Cycle error: {exc}", exc_info=True)

        if config.CRYPTO_REOPT_INTERVAL_MIN > 0:
            logger.info(
                f"[CRYPTO] Sleeping {config.CRYPTO_REOPT_INTERVAL_MIN}m until next cycle..."
            )
            time.sleep(config.CRYPTO_REOPT_INTERVAL_MIN * 60)
        else:
            logger.info("[CRYPTO] Looping immediately (CRYPTO_REOPT_INTERVAL_MIN=0)...")


def run_workflow(mode: str = "stock") -> None:
    """
    Dispatch to the appropriate workflow scheduler.

    mode : "stock"  — technical + fundamental + screen (market-hours gated)
           "crypto" — technical optimize only, 24/7
           "all"    — stock workflow then crypto workflow (sequential)
    """
    if mode == "crypto":
        run_workflow_crypto()
        return
    if mode == "all":
        import threading
        t_crypto = threading.Thread(target=run_workflow_crypto, daemon=True, name="crypto-wf")
        t_crypto.start()
        # Fall through to run the stock workflow on the main thread
        _run_workflow_stock()
        return
    _run_workflow_stock()


def _run_workflow_stock() -> None:
    """
    Full stock scheduler — runs technical and fundamental agents on their cadences.

    Cadences
    --------
    Every 30 min (market hours) : Technical agent — monitor + re-optimize if degraded
    Weekly                      : Universe screen → update watchlist
    Monthly                     : Fundamental model re-optimization

    State tracked in registry/schedule.json.
    """
    logger.info("=" * 60)
    logger.info("  TRADING AGENT WORKFLOW SCHEDULER")
    logger.info("=" * 60)
    logger.info(f"  Universe tier   : {config.UNIVERSE_TIER}")
    logger.info(f"  Screen cadence  : every {config.SCREEN_INTERVAL_DAYS} days")
    logger.info(f"  Reopt cadence   : every {config.REOPT_INTERVAL_DAYS} days")
    logger.info(f"  Check cadence   : every {config.CHECK_INTERVAL_MIN} min (market hours)")
    logger.info("=" * 60)

    while True:
        schedule = _load_schedule()
        now      = datetime.now(timezone.utc)

        # ------------------------------------------------------------------
        # WEEKLY: Universe screen → update watchlist
        # ------------------------------------------------------------------
        days_since_screen = _days_since(schedule.get("last_screen"))
        if days_since_screen >= config.SCREEN_INTERVAL_DAYS:
            logger.info(f"[SCREEN] {days_since_screen:.1f} days since last screen "
                        f"(threshold={config.SCREEN_INTERVAL_DAYS}d) — running...")
            try:
                passed = run_screen(save=True)
                schedule["last_screen"]  = _now_iso()
                schedule["watchlist"]    = passed
                _save_schedule(schedule)
                logger.info(f"[SCREEN] Done — watchlist updated: {passed}")
            except Exception as exc:
                logger.error(f"[SCREEN] Failed: {exc}", exc_info=True)

        # ------------------------------------------------------------------
        # MONTHLY: Fundamental model re-optimization
        # ------------------------------------------------------------------
        days_since_reopt = _days_since(schedule.get("last_fund_reopt"))
        if days_since_reopt >= config.REOPT_INTERVAL_DAYS:
            logger.info(f"[FUND-OPT] {days_since_reopt:.1f} days since last fundamental reopt "
                        f"(threshold={config.REOPT_INTERVAL_DAYS}d) — running...")
            try:
                from tools.fundamental_optimize_tool import optimize_fundamental
                watchlist = _load_watchlist()
                # Train on up to 5 tickers for robustness
                train_tickers = watchlist[:5] if watchlist else config.WATCHLIST[:3]
                logger.info(f"[FUND-OPT] Training on: {train_tickers}")
                result = optimize_fundamental(
                    ticker=train_tickers[0],
                    extra_train_tickers=train_tickers[1:] if len(train_tickers) > 1 else None,
                    lookback_years=3.0,
                    generations=100,
                    popsize=40,
                )
                if result.get("sharpe", 0) >= config.FUND_GATE_MIN_SHARPE:
                    import shutil
                    config.FUNDAMENTAL_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(config.FUNDAMENTAL_PARAMS_FILE, "w") as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"[FUND-OPT] Deployed — Sharpe={result['sharpe']:.3f}")
                    schedule["last_fund_reopt"] = _now_iso()
                    _save_schedule(schedule)
                else:
                    logger.warning(f"[FUND-OPT] Rejected — Sharpe={result.get('sharpe'):.3f} "
                                   f"< {config.FUND_GATE_MIN_SHARPE}")
            except Exception as exc:
                logger.error(f"[FUND-OPT] Failed: {exc}", exc_info=True)

        # ------------------------------------------------------------------
        # EVERY CHECK_INTERVAL_MIN (market hours): Technical agent
        # ------------------------------------------------------------------
        if _is_market_hours():
            logger.info("[TECH] Market open — running technical agent...")
            try:
                run_once("technical")
                schedule["last_tech_check"] = _now_iso()
                _save_schedule(schedule)
            except Exception as exc:
                logger.error(f"[TECH] Agent error: {exc}", exc_info=True)
        else:
            next_check_min = config.CHECK_INTERVAL_MIN
            logger.info(f"[TECH] Market closed — sleeping {next_check_min}m...")

        time.sleep(config.CHECK_INTERVAL_MIN * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_review_only() -> None:
    """
    Dry-run adversarial review of the currently deployed params.
    Runs backtest → Proposer → Skeptic → Judge and prints the verdict.
    Does NOT deploy anything regardless of verdict.
    """
    from tools.adversarial_review_tool import review_deployed_params
    import json as _json

    crypto = config.TRADING_MODE == "crypto"
    logger.info("Running adversarial review on deployed params (dry run — no deployment)...")
    verdict = review_deployed_params(crypto=crypto)

    logger.info("=" * 60)
    logger.info("ADVERSARIAL REVIEW RESULT")
    logger.info("=" * 60)
    logger.info("Verdict    : %s", verdict.outcome)
    logger.info("Confidence : %s", f"{verdict.confidence:.2f}" if verdict.confidence is not None else "n/a")
    for risk in verdict.raw_artifacts.get("verdict_data", {}).get("key_risks", []):
        logger.info("Risk       : %s", risk)
    logger.info("Reasoning  : %s", verdict.reason)
    logger.info("-" * 60)
    logger.info("PROPOSER ARGUMENT:")
    logger.info(verdict.raw_artifacts.get("proposer_argument", ""))
    logger.info("-" * 60)
    logger.info("SKEPTIC REBUTTAL:")
    logger.info(verdict.raw_artifacts.get("skeptic_rebuttal", ""))
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Trading Agent Orchestrator")
    parser.add_argument("--agent", default="technical",
                        choices=["technical", "fundamental", "sentiment"],
                        help="Agent to run (default: technical)")
    parser.add_argument("--loop",               action="store_true",
                        help="Run technical agent in a continuous market-hours loop")
    parser.add_argument("--workflow",
                        nargs="?", const="stock", default=None,
                        choices=["stock", "crypto", "all"],
                        metavar="MODE",
                        help="Full scheduler: stock (default), crypto (24/7), or all")
    parser.add_argument("--screen",              action="store_true",
                        help="Run universe screen → fundamental analysis → save watchlist")
    parser.add_argument("--check",               action="store_true",
                        help="Dry-run: validate config and paths")
    parser.add_argument("--backtest-only",       action="store_true",
                        help="Backtest deployed params without deploying")
    parser.add_argument("--optimize-and-deploy", action="store_true",
                        help="Full technical optimize → validate → deploy cycle")
    parser.add_argument("--review-only",          action="store_true",
                        help="Dry-run adversarial review of currently deployed params (no deployment)")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if _check() else 1)
    elif args.review_only:
        _run_review_only()
    elif args.screen:
        run_screen(save=True)
    elif args.workflow is not None:
        run_workflow(mode=args.workflow)
    elif args.backtest_only:
        backtest_only()
    elif args.optimize_and_deploy:
        optimize_and_deploy()
    elif args.loop:
        run_loop(args.agent)
    else:
        run_once(args.agent)


if __name__ == "__main__":
    main()
