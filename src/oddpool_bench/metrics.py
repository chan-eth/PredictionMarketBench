"""
Metrics computation for benchmark evaluation.
"""

from dataclasses import dataclass
from typing import Optional
import math

from .types import EquitySnapshot, EpisodeResult


@dataclass
class Metrics:
    """Computed metrics for an episode."""
    
    total_pnl_cents: int
    total_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    total_contracts_traded: int
    total_notional_cents: int
    total_fees_cents: int
    total_slippage_cents: int
    fill_ratio: float


def compute_max_drawdown(equity_curve: list[EquitySnapshot]) -> float:
    """
    Compute maximum drawdown as a percentage.
    
    Drawdown = (peak - current) / peak
    """
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0].equity_cents
    max_dd = 0.0
    
    for snap in equity_curve:
        if snap.equity_cents > peak:
            peak = snap.equity_cents
        if peak > 0:
            dd = (peak - snap.equity_cents) / peak
            max_dd = max(max_dd, dd)
    
    return max_dd


def compute_sharpe_ratio(
    equity_curve: list[EquitySnapshot],
    sampling_interval_seconds: int = 60,
    risk_free_rate: float = 0.0,
) -> Optional[float]:
    """
    Compute annualized Sharpe ratio from equity curve.
    
    Uses the standard formula: Sharpe = (E[R] - Rf) / σ(R) * √(N)
    where:
    - E[R] is the mean return per period
    - Rf is the risk-free rate per period
    - σ(R) is the standard deviation of returns (sample std, using n-1)
    - N is the number of periods per year (for annualization)
    
    Args:
        equity_curve: List of equity snapshots
        sampling_interval_seconds: Interval for computing returns (default 60s)
        risk_free_rate: Annualized risk-free rate (default 0)
        
    Returns:
        Annualized Sharpe ratio, or None if insufficient data
    """
    if len(equity_curve) < 2:
        return None
    
    # Sample equity at regular intervals
    sampled = []
    last_ts = None
    
    for snap in equity_curve:
        if last_ts is None:
            sampled.append(snap.equity_cents)
            last_ts = snap.ts
        elif (snap.ts - last_ts).total_seconds() >= sampling_interval_seconds:
            sampled.append(snap.equity_cents)
            last_ts = snap.ts
    
    if len(sampled) < 3:  # Need at least 3 points for meaningful stats
        return None
    
    # Compute returns (percentage change between periods)
    returns = []
    for i in range(1, len(sampled)):
        if sampled[i - 1] > 0:
            ret = (sampled[i] - sampled[i - 1]) / sampled[i - 1]
            returns.append(ret)
    
    if len(returns) < 2:
        return None
    
    n = len(returns)
    
    # Compute mean return
    mean_return = sum(returns) / n
    
    # Compute sample standard deviation (using n-1 for unbiased estimator)
    variance = sum((r - mean_return) ** 2 for r in returns) / (n - 1)
    std_return = math.sqrt(variance) if variance > 0 else 0
    
    if std_return == 0:
        # No volatility - can't compute meaningful Sharpe
        return None
    
    # Annualization factor
    # Intervals per year ≈ seconds_per_year / sampling_interval
    seconds_per_year = 365.25 * 24 * 3600
    intervals_per_year = seconds_per_year / sampling_interval_seconds
    
    # Convert annual risk-free rate to per-interval rate
    rf_per_interval = risk_free_rate / intervals_per_year
    
    # Sharpe ratio: (mean_return - rf) / std * sqrt(intervals_per_year)
    # This annualizes the ratio by multiplying by sqrt of periods per year
    sharpe = (mean_return - rf_per_interval) / std_return * math.sqrt(intervals_per_year)
    
    return sharpe


def compute_metrics(
    initial_equity: int,
    final_equity: int,
    equity_curve: list[EquitySnapshot],
    total_contracts: int,
    total_notional: int,
    total_fees: int,
    total_slippage: int,
    fill_ratio: float,
) -> Metrics:
    """
    Compute all metrics for an episode.
    """
    pnl = final_equity - initial_equity
    pnl_pct = pnl / initial_equity if initial_equity > 0 else 0.0
    
    return Metrics(
        total_pnl_cents=pnl,
        total_pnl_pct=pnl_pct,
        max_drawdown_pct=compute_max_drawdown(equity_curve),
        sharpe_ratio=compute_sharpe_ratio(equity_curve),
        total_contracts_traded=total_contracts,
        total_notional_cents=total_notional,
        total_fees_cents=total_fees,
        total_slippage_cents=total_slippage,
        fill_ratio=fill_ratio,
    )


def aggregate_metrics(results: list[EpisodeResult]) -> dict:
    """
    Aggregate metrics across multiple episodes.
    
    Returns dict with mean and median for each metric.
    """
    if not results:
        return {}
    
    def mean(values):
        return sum(values) / len(values)
    
    def median(values):
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        return sorted_vals[n // 2]
    
    pnls = [r.total_pnl_cents for r in results]
    pnl_pcts = [r.total_pnl_pct for r in results]
    drawdowns = [r.max_drawdown_pct for r in results]
    sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio is not None]
    fill_ratios = [r.fill_ratio for r in results]
    
    return {
        "n_episodes": len(results),
        "total_pnl_cents": {
            "mean": mean(pnls),
            "median": median(pnls),
            "sum": sum(pnls),
        },
        "total_pnl_pct": {
            "mean": mean(pnl_pcts),
            "median": median(pnl_pcts),
        },
        "max_drawdown_pct": {
            "mean": mean(drawdowns),
            "median": median(drawdowns),
            "max": max(drawdowns),
        },
        "sharpe_ratio": {
            "mean": mean(sharpes) if sharpes else None,
            "median": median(sharpes) if sharpes else None,
        },
        "fill_ratio": {
            "mean": mean(fill_ratios),
            "median": median(fill_ratios),
        },
    }
