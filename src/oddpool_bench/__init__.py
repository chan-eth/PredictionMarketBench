"""
Oddpool PredictionMarketBench - A benchmark for prediction market trading agents.

Replays real Kalshi episodes and evaluates agent trading performance under realistic
execution constraints.
"""

__version__ = "0.2.0"

from .types import (
    Side,
    Action,
    OrderType,
    TimeInForce,
    FillType,
    Order,
    Fill,
    RestingOrder,
    OrderResult,
    Position,
    MarketInfo,
    OrderbookLevel,
    OrderbookSnapshot,
    SettlementResult,
    EpisodeMetadata,
    EquitySnapshot,
    EpisodeResult,
)
from .agent import Agent, AgentContext
from .simulator import Simulator, SimulatorConfig
from .harness import BenchmarkHarness, BenchmarkResult
from .fees import FeeModel, KalshiOct2025FeeModel, get_fee_model
from .maker_queue import MakerQueueManager, LevelQueue
from .data import TradePrint, load_trades_data

__all__ = [
    # Types
    "Side",
    "Action",
    "OrderType",
    "TimeInForce",
    "FillType",
    "Order",
    "Fill",
    "RestingOrder",
    "OrderResult",
    "Position",
    "MarketInfo",
    "OrderbookLevel",
    "OrderbookSnapshot",
    "SettlementResult",
    "EpisodeMetadata",
    "EquitySnapshot",
    "EpisodeResult",
    # Core
    "Agent",
    "AgentContext",
    "Simulator",
    "SimulatorConfig",
    "BenchmarkHarness",
    "BenchmarkResult",
    # Fees
    "FeeModel",
    "KalshiOct2025FeeModel",
    "get_fee_model",
    # Maker
    "MakerQueueManager",
    "LevelQueue",
    "TradePrint",
    "load_trades_data",
]
