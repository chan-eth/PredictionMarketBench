"""
Oddpool PredictionMarketBench - A benchmark for prediction market trading agents.

Replays real Kalshi episodes and evaluates agent trading performance under realistic
execution constraints.
"""

__version__ = "0.1.0"

from .types import (
    Side,
    Action,
    OrderType,
    TimeInForce,
    Order,
    Fill,
    OrderResult,
    Position,
    MarketInfo,
    OrderbookLevel,
    OrderbookSnapshot,
    SettlementResult,
    EpisodeMetadata,
)
from .agent import Agent, AgentContext
from .simulator import Simulator, SimulatorConfig
from .harness import BenchmarkHarness, BenchmarkResult

__all__ = [
    # Types
    "Side",
    "Action",
    "OrderType",
    "TimeInForce",
    "Order",
    "Fill",
    "OrderResult",
    "Position",
    "MarketInfo",
    "OrderbookLevel",
    "OrderbookSnapshot",
    "SettlementResult",
    "EpisodeMetadata",
    # Core
    "Agent",
    "AgentContext",
    "Simulator",
    "SimulatorConfig",
    "BenchmarkHarness",
    "BenchmarkResult",
]
