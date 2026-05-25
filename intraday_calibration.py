"""Calibration backtest for the intraday lognormal model.

Fetches recent 1-min OHLCV candles from Binance.US and tests whether
the lognormal probability model is well-calibrated on historical data.

No Kalshi credentials needed -- uses only the public Binance.US API.

What it measures
----------------
- Calibration: when the model says 70%, does YES resolve ~70% of the time?
- Brier score: mean squared error of probability predictions (lower = better)
- Edge analysis: at each edge threshold, what is the actual win rate?
- Static vs dynamic-vol comparison: does realized vol improve calibration?

Usage
-----
    python backtest-framework/intraday_calibration.py
    python backtest-framework/intraday_calibration.py --symbol ETHUSDT --days 7
    python backtest-framework/intraday_calibration.py --symbol BTCUSDT --days 3 --min-edge 0.03
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from typing import NamedTuple

import httpx

# -- Math (inlined so script runs standalone without PYTHONPATH setup) ---------

MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0
WINDOW_MINUTES = 15       # Kalshi KXBTC15M settlement window
VOL_LOOKBACK = 30         # candles of realized-vol history
STATIC_SIGMA = {
    "BTCUSDT": 0.75,
    "ETHUSDT": 1.00,
    "SOLUSDT": 1.20,
}
# Keep strikes tight -- at 15-min horizon sigma_T ~0.4%, so ±5% is >10 sigma
# (essentially certain). Only near-ATM strikes produce meaningful probabilities.
STRIKES_MONEYNESS = [-0.010, -0.006, -0.003, -0.001, 0.0, 0.001, 0.003, 0.006, 0.010]
VOL_FLOOR = 0.15
VOL_CAP   = 5.00


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def lognormal_yes_prob(spot: float, strike: float, sigma_annual: float, minutes: float) -> float:
    if minutes <= 0 or spot <= 0 or strike <= 0 or sigma_annual <= 0:
        return 0.0
    T = minutes / MINUTES_PER_YEAR
    sigma_T = sigma_annual * math.sqrt(T)
    if sigma_T < 1e-12:
        return 1.0 if spot > strike else 0.0
    d2 = math.log(spot / strike) / sigma_T
    return _norm_cdf(d2)


def realized_vol_annual(closes: list[float]) -> float | None:
    if len(closes) < 3:
        return None
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            return None
        returns.append(math.log(closes[i] / closes[i - 1]))
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return None
    return math.sqrt(var) * math.sqrt(MINUTES_PER_YEAR)


# -- Binance.US data fetch -----------------------------------------------------

BINANCE_KLINES = "https://api.binance.us/api/v3/klines"


def fetch_candles(symbol: str, days: int) -> list[dict]:
    """Fetch `days` worth of 1-min candles from Binance.US (public, no auth).

    Returns list of dicts: {open_ms, close_ms, open, high, low, close, volume}
    """
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    candles  = []
    batch_ms = start_ms
    print(f"Fetching {days}d of {symbol} 1-min candles from Binance.US ...", flush=True)
    with httpx.Client(timeout=30.0) as client:
        while batch_ms < end_ms:
            resp = client.get(BINANCE_KLINES, params={
                "symbol": symbol, "interval": "1m",
                "startTime": str(batch_ms), "limit": "1000",
            })
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for k in batch:
                candles.append({
                    "open_ms":  int(k[0]),
                    "close_ms": int(k[6]),
                    "open":  float(k[1]),
                    "high":  float(k[2]),
                    "low":   float(k[3]),
                    "close": float(k[4]),
                })
            batch_ms = int(batch[-1][6]) + 1  # next candle after last close_ms
            if len(batch) < 1000:
                break
    print(f"  Got {len(candles):,} candles ({len(candles)/60:.1f}h of data)", flush=True)
    return candles


# -- Backtest engine -----------------------------------------------------------

class Prediction(NamedTuple):
    static_prob:   float
    dynamic_prob:  float
    market_approx: float   # ATM proxy = 0.5 for ATM, varies for off-strikes
    outcome:       int     # 1 = YES resolved, 0 = NO resolved
    moneyness:     float   # (strike - spot) / spot
    edge_static:   float   # abs(static_prob - market_approx)
    edge_dynamic:  float   # abs(dynamic_prob - market_approx)


def run_backtest(candles: list[dict], symbol: str) -> list[Prediction]:
    """Slide a 15-min window over the candle history.

    For each window:
      - spot   = close of candle at window start (index i)
      - outcome = 1 if close at window end (index i+15) > strike
      - realized vol computed from candles[i-VOL_LOOKBACK : i]
      - static vol from STATIC_SIGMA[symbol]
    """
    static_sigma = STATIC_SIGMA.get(symbol, 0.80)
    preds: list[Prediction] = []
    n = len(candles)
    step = 1  # slide every minute for maximum data

    for i in range(VOL_LOOKBACK, n - WINDOW_MINUTES):
        spot    = candles[i]["close"]
        outcome_close = candles[i + WINDOW_MINUTES]["close"]

        # Realized vol from the lookback window
        lookback_closes = [c["close"] for c in candles[i - VOL_LOOKBACK: i + 1]]
        rv = realized_vol_annual(lookback_closes)
        dynamic_sigma = max(VOL_FLOOR, min(rv, VOL_CAP)) if rv is not None else static_sigma

        for m in STRIKES_MONEYNESS:
            strike  = spot * (1.0 + m)
            outcome = 1 if outcome_close > strike else 0

            sp = lognormal_yes_prob(spot, strike, static_sigma,  WINDOW_MINUTES)
            dp = lognormal_yes_prob(spot, strike, dynamic_sigma, WINDOW_MINUTES)

            # Market approximation: we don't have live Kalshi market prices,
            # so we use the lognormal with static vol as a stand-in for "market price".
            # This tests dynamic vol vs static, and calibration vs actual outcomes.
            market_approx = sp

            preds.append(Prediction(
                static_prob   = sp,
                dynamic_prob  = dp,
                market_approx = market_approx,
                outcome       = outcome,
                moneyness     = m,
                edge_static   = abs(dp - sp),   # how much dynamic diverges from static
                edge_dynamic  = abs(dp - market_approx),
            ))

    return preds


# -- Analysis -----------------------------------------------------------------

def calibration_table(preds: list[Prediction], use_dynamic: bool) -> None:
    """Print reliability table: predicted prob bin --> actual frequency."""
    bins: dict[int, list[int]] = defaultdict(list)
    for p in preds:
        prob = p.dynamic_prob if use_dynamic else p.static_prob
        bucket = int(prob * 10)  # 0-9 --> 0-10%, 10-20%, ..., 90-100%
        bucket = min(bucket, 9)
        bins[bucket].append(p.outcome)

    label = "Dynamic vol" if use_dynamic else "Static vol "
    print(f"\n  {label} -- reliability table (lower Brier = better)")
    print(f"  {'Pred %':>8}  {'Actual %':>9}  {'N':>6}  {'Brier contrib':>13}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*13}")
    total_brier = 0.0
    total_n = 0
    for bucket in range(10):
        outcomes = bins.get(bucket, [])
        if not outcomes:
            continue
        pred_mid = (bucket + 0.5) / 10.0
        actual   = sum(outcomes) / len(outcomes)
        brier    = sum((pred_mid - o) ** 2 for o in outcomes) / len(outcomes)
        total_brier += brier * len(outcomes)
        total_n     += len(outcomes)
        bar = "#" * int(actual * 20)
        print(f"  {pred_mid:>7.0%}  {actual:>9.1%}  {len(outcomes):>6,}  {brier:>13.4f}  {bar}")
    if total_n:
        print(f"  {'TOTAL':>8}  {'':>9}  {total_n:>6,}  {total_brier/total_n:>13.4f}  <-- Brier score")


def edge_analysis(preds: list[Prediction], thresholds: list[float]) -> None:
    """For each edge threshold, show win rate and expected value."""
    print("\n  Edge analysis (dynamic vol vs static vol as proxy market price)")
    print(f"  {'MinEdge':>8}  {'Signals':>8}  {'WinRate':>8}  {'ExpVal':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for thresh in thresholds:
        selected = [p for p in preds if p.edge_static >= thresh]
        if not selected:
            print(f"  {thresh:>8.1%}  {'0':>8}  {'--':>8}  {'--':>8}")
            continue
        wins = sum(
            1 for p in selected
            if (p.dynamic_prob > p.market_approx and p.outcome == 1) or
               (p.dynamic_prob < p.market_approx and p.outcome == 0)
        )
        win_rate = wins / len(selected)
        # Expected value: bet $1 on model direction at market price, collect $1/mkt if win
        ev = sum(
            (1.0 / p.market_approx - 1.0) if (p.dynamic_prob > p.market_approx and p.outcome == 1)
            else (1.0 / (1.0 - p.market_approx) - 1.0) if (p.dynamic_prob < p.market_approx and p.outcome == 0)
            else -1.0
            for p in selected
        ) / len(selected)
        print(f"  {thresh:>8.1%}  {len(selected):>8,}  {win_rate:>8.1%}  {ev:>+8.3f}")


def vol_comparison(preds: list[Prediction]) -> None:
    """Compare Brier scores of static vs dynamic vol."""
    def brier(probs_outcomes):
        return sum((p - o) ** 2 for p, o in probs_outcomes) / len(probs_outcomes)

    static_b  = brier([(p.static_prob,  p.outcome) for p in preds])
    dynamic_b = brier([(p.dynamic_prob, p.outcome) for p in preds])
    delta = static_b - dynamic_b
    improvement = delta / static_b * 100 if static_b else 0

    print(f"\n  Vol comparison (Brier score, lower = better)")
    print(f"  Static  vol Brier: {static_b:.5f}")
    print(f"  Dynamic vol Brier: {dynamic_b:.5f}")
    sign = "+" if delta >= 0 else ""
    print(f"  Dynamic improvement: {sign}{delta:.5f} ({sign}{improvement:.1f}%)")
    if delta > 0:
        print("  --> Dynamic vol is BETTER calibrated")
    elif delta < 0:
        print("  --> Dynamic vol is WORSE (static is better for this dataset)")
    else:
        print("  --> No difference")


def moneyness_breakdown(preds: list[Prediction]) -> None:
    """Show calibration quality broken down by strike moneyness."""
    groups: dict[float, list[Prediction]] = defaultdict(list)
    for p in preds:
        groups[p.moneyness].append(p)

    print(f"\n  Moneyness breakdown (dynamic vol Brier per strike bucket)")
    print(f"  {'Moneyness':>10}  {'N':>7}  {'ActualYES%':>11}  {'Brier':>8}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*11}  {'-'*8}")
    for m in sorted(groups):
        g = groups[m]
        actual = sum(p.outcome for p in g) / len(g)
        brier  = sum((p.dynamic_prob - p.outcome) ** 2 for p in g) / len(g)
        print(f"  {m:>+10.1%}  {len(g):>7,}  {actual:>11.1%}  {brier:>8.4f}")


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday lognormal calibration backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance.US symbol (default: BTCUSDT)")
    parser.add_argument("--days",   type=int, default=3, help="Days of history to fetch (default: 3)")
    parser.add_argument("--min-edge", type=float, default=0.03, dest="min_edge",
                        help="Minimum edge threshold for live trading (default: 0.03)")
    args = parser.parse_args()

    candles = fetch_candles(args.symbol, args.days)
    if len(candles) < VOL_LOOKBACK + WINDOW_MINUTES + 10:
        print("Not enough candle data -- try --days 2 or higher.")
        sys.exit(1)

    print(f"\nRunning calibration backtest on {args.symbol} ({args.days}d) ...")
    preds = run_backtest(candles, args.symbol)
    print(f"Generated {len(preds):,} predictions across {len(STRIKES_MONEYNESS)} strike levels")

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  CALIBRATION RESULTS -- {args.symbol} ({args.days}d, {len(preds):,} obs)")
    print(sep)

    calibration_table(preds, use_dynamic=False)
    calibration_table(preds, use_dynamic=True)
    vol_comparison(preds)
    moneyness_breakdown(preds)
    edge_analysis(preds, [0.01, 0.02, 0.03, 0.05, 0.08, 0.10])

    print(f"\n{sep}")
    print(f"  RECOMMENDATION")
    print(sep)

    # Simple heuristic: is win rate at configured edge threshold > 55%?
    selected = [p for p in preds if p.edge_static >= args.min_edge]
    if not selected:
        print(f"  No predictions found with edge >= {args.min_edge:.0%} -- threshold may be too high.")
    else:
        wins = sum(
            1 for p in selected
            if (p.dynamic_prob > p.market_approx and p.outcome == 1) or
               (p.dynamic_prob < p.market_approx and p.outcome == 0)
        )
        win_rate = wins / len(selected)
        print(f"  At edge >= {args.min_edge:.0%}: {len(selected):,} signals, win rate = {win_rate:.1%}")
        if win_rate >= 0.55:
            print(f"  Model appears VIABLE at this threshold (>55% win rate).")
            print(f"  Consider running dry-run for 1+ week to validate with live Kalshi prices.")
        else:
            print(f"  Win rate below 55% -- model needs tuning before going live.")
            print(f"  Try a higher edge threshold (e.g. --min-edge 0.05) or more data (--days 7).")
    print()


if __name__ == "__main__":
    main()
