"""
Core type definitions for the Oddpool PredictionMarketBench.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(Enum):
    """Contract side: YES or NO."""
    YES = "yes"
    NO = "no"


class Action(Enum):
    """Order action: BUY or SELL."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(Enum):
    """Time in force for orders."""
    IOC = "ioc"  # Immediate or cancel (taker only)
    GTC = "gtc"  # Good til canceled (can rest as maker)
    GTD = "gtd"  # Good til date (not supported yet)
    POST_ONLY = "post_only"  # Must rest as maker, reject if would cross


class FillType(Enum):
    """Whether a fill was maker or taker."""
    TAKER = "taker"
    MAKER = "maker"


@dataclass
class OrderbookLevel:
    """A single level in the orderbook."""
    price_cents: int  # Price in cents (1-99)
    size: int  # Number of contracts


@dataclass
class OrderbookSnapshot:
    """Snapshot of orderbook for a single ticker at a point in time."""
    ts: datetime
    sequence_id: int
    ticker: str
    yes_bids: list[OrderbookLevel]  # Sorted best (highest) first
    no_bids: list[OrderbookLevel]   # Sorted best (highest) first
    
    def get_yes_asks(self) -> list[OrderbookLevel]:
        """
        Derive YES asks from NO bids.
        yes_ask_price = 100 - no_bid_price
        Sorted best (lowest) first.
        """
        asks = [
            OrderbookLevel(price_cents=100 - level.price_cents, size=level.size)
            for level in self.no_bids
        ]
        # Sort by price ascending (best ask = lowest price first)
        asks.sort(key=lambda x: x.price_cents)
        return asks
    
    def get_no_asks(self) -> list[OrderbookLevel]:
        """
        Derive NO asks from YES bids.
        no_ask_price = 100 - yes_bid_price
        Sorted best (lowest) first.
        """
        asks = [
            OrderbookLevel(price_cents=100 - level.price_cents, size=level.size)
            for level in self.yes_bids
        ]
        # Sort by price ascending (best ask = lowest price first)
        asks.sort(key=lambda x: x.price_cents)
        return asks
    
    @property
    def yes_best_bid(self) -> Optional[int]:
        """Best YES bid price in cents, or None if no bids."""
        return self.yes_bids[0].price_cents if self.yes_bids else None
    
    @property
    def yes_best_ask(self) -> Optional[int]:
        """Best YES ask price in cents, or None if no asks."""
        asks = self.get_yes_asks()
        return asks[0].price_cents if asks else None
    
    @property
    def no_best_bid(self) -> Optional[int]:
        """Best NO bid price in cents, or None if no bids."""
        return self.no_bids[0].price_cents if self.no_bids else None
    
    @property
    def no_best_ask(self) -> Optional[int]:
        """Best NO ask price in cents, or None if no asks."""
        asks = self.get_no_asks()
        return asks[0].price_cents if asks else None
    
    @property
    def yes_mid(self) -> Optional[float]:
        """Mid price for YES side in cents, or None if no market."""
        if self.yes_best_bid is not None and self.yes_best_ask is not None:
            return (self.yes_best_bid + self.yes_best_ask) / 2
        return None
    
    @property
    def no_mid(self) -> Optional[float]:
        """Mid price for NO side in cents, or None if no market."""
        if self.no_best_bid is not None and self.no_best_ask is not None:
            return (self.no_best_bid + self.no_best_ask) / 2
        return None


@dataclass
class Order:
    """An order to be placed."""
    ticker: str
    side: Side  # YES or NO
    action: Action  # BUY or SELL
    order_type: OrderType
    count: int  # Number of contracts
    limit_price_cents: Optional[int] = None  # Required for limit orders
    time_in_force: TimeInForce = TimeInForce.IOC
    order_id: Optional[str] = None  # Optional client order ID for tracking
    
    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.limit_price_cents is None:
            raise ValueError("Limit orders require limit_price_cents")
        if self.count <= 0:
            raise ValueError("Order count must be positive")
    
    def to_canonical_bid(self) -> tuple[Side, int]:
        """
        Convert order to canonical bid representation.
        
        Returns (contract_side, price_cents) where we're placing a bid
        on contract_side at price_cents.
        
        Mapping:
        - Buy YES @ p → YES bid @ p
        - Sell YES @ p → NO bid @ (100-p)
        - Buy NO @ p → NO bid @ p  
        - Sell NO @ p → YES bid @ (100-p)
        """
        if self.action == Action.BUY:
            return (self.side, self.limit_price_cents)
        else:
            # Selling = bidding on opposite side at flipped price
            opposite = Side.NO if self.side == Side.YES else Side.YES
            return (opposite, 100 - self.limit_price_cents)


@dataclass
class Fill:
    """A single fill within an order execution."""
    price_cents: int
    size: int
    fee_cents: int
    fill_type: FillType = FillType.TAKER  # Taker or maker
    fill_ts: Optional[datetime] = None  # When fill occurred (for maker fills)


@dataclass
class RestingOrder:
    """A resting (maker) order in the queue."""
    order_id: str
    original_order: Order
    contract_side: Side  # Canonical: YES or NO bid
    price_cents: int  # Canonical bid price
    remaining_count: int
    placed_ts: datetime
    env_ahead: int  # Environment size ahead when placed
    fills: list[Fill] = field(default_factory=list)


@dataclass
class OrderResult:
    """Result of an order execution."""
    order: Order
    ts: datetime
    filled_count: int
    fills: list[Fill]
    total_cost_cents: int  # Total cost including fees (negative for sales)
    total_fees_cents: int
    rejected: bool = False
    reject_reason: Optional[str] = None
    resting_order: Optional[RestingOrder] = None  # For GTC orders that rest
    
    @property
    def average_fill_price(self) -> Optional[float]:
        """Average fill price in cents, or None if no fills."""
        if self.filled_count == 0:
            return None
        total_notional = sum(f.price_cents * f.size for f in self.fills)
        return total_notional / self.filled_count


@dataclass
class Position:
    """Position in a single ticker."""
    ticker: str
    yes_contracts: int = 0
    no_contracts: int = 0
    yes_avg_cost_cents: float = 0.0  # Average cost per YES contract
    no_avg_cost_cents: float = 0.0   # Average cost per NO contract
    realized_pnl_cents: int = 0


@dataclass
class MarketInfo:
    """Information about a market/ticker."""
    ticker: str
    event_slug: str
    yes_best_bid: Optional[int]
    yes_best_ask: Optional[int]
    no_best_bid: Optional[int]
    no_best_ask: Optional[int]
    status: str  # "active", "halted", "settled"
    time_to_close_seconds: Optional[float]  # Seconds until settlement, if known


@dataclass
class SettlementResult:
    """Settlement outcome for a ticker."""
    ticker: str
    result: str  # "YES" or "NO"
    settled_ts: datetime


@dataclass
class EpisodeMetadata:
    """Metadata for a benchmark episode."""
    episode_id: str
    event_slug: str
    tickers: list[str]
    start_ts: datetime
    end_ts: datetime
    initial_bankroll_cents: int
    fee_model_version: str
    execution_mode: str  # "taker_only", "maker_taker"
    observation_depth: int  # Number of orderbook levels (-1 = full depth)
    description: Optional[str] = None
    has_trades_tape: bool = False  # Whether trade data is available for maker fills
    
    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeMetadata":
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            episode_id=d["episode_id"],
            event_slug=d["event_slug"],
            tickers=d["tickers"],
            start_ts=datetime.fromisoformat(d["start_ts"]),
            end_ts=datetime.fromisoformat(d["end_ts"]),
            initial_bankroll_cents=d["initial_bankroll_cents"],
            fee_model_version=d["fee_model_version"],
            execution_mode=d["execution_mode"],
            observation_depth=d["observation_depth"],
            description=d.get("description"),
            has_trades_tape=d.get("has_trades_tape", False),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "event_slug": self.event_slug,
            "tickers": self.tickers,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat(),
            "initial_bankroll_cents": self.initial_bankroll_cents,
            "fee_model_version": self.fee_model_version,
            "execution_mode": self.execution_mode,
            "observation_depth": self.observation_depth,
            "description": self.description,
            "has_trades_tape": self.has_trades_tape,
        }


@dataclass
class EquitySnapshot:
    """Snapshot of portfolio equity at a point in time."""
    ts: datetime
    cash_cents: int
    position_value_cents: int  # Mark-to-market value of positions
    equity_cents: int  # cash + position_value
    
    
@dataclass
class EpisodeResult:
    """Results from running an episode."""
    episode_id: str
    initial_equity_cents: int
    final_equity_cents: int
    total_pnl_cents: int
    total_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    total_contracts_traded: int
    total_notional_cents: int
    total_fees_cents: int
    total_slippage_cents: int
    fill_ratio: float  # Fraction of requested contracts filled
    equity_curve: list[EquitySnapshot]
    order_results: list[OrderResult]
    settlements: dict[str, SettlementResult]
