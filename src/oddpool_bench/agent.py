"""
Agent interface for the benchmark.

Defines the abstract base class that trading agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .types import (
    Order,
    Position,
    MarketInfo,
    OrderbookSnapshot,
    OrderbookLevel,
)


class AgentContext:
    """
    Context provided to agents with tool access.
    
    This wraps the simulator state and provides Kalshi-like API tools
    for agents to query market state and place orders.
    """
    
    def __init__(
        self,
        current_ts,
        orderbooks: dict[str, OrderbookSnapshot],
        positions: dict[str, Position],
        cash_cents: int,
        equity_cents: int,
        tickers: list[str],
        event_slug: str,
        end_ts,
        place_order_callback,
        observation_depth: int = 5,
    ):
        self._current_ts = current_ts
        self._orderbooks = orderbooks
        self._positions = positions
        self._cash_cents = cash_cents
        self._equity_cents = equity_cents
        self._tickers = tickers
        self._event_slug = event_slug
        self._end_ts = end_ts
        self._place_order = place_order_callback
        self._observation_depth = observation_depth
        self._tool_calls = 0
        self._max_tool_calls = 100  # Budget per step
    
    def _check_tool_budget(self):
        """Check if tool call budget is exceeded."""
        self._tool_calls += 1
        if self._tool_calls > self._max_tool_calls:
            raise RuntimeError(
                f"Tool call budget exceeded ({self._max_tool_calls} calls per step)"
            )
    
    # ============= Agent Tools (Kalshi-like API) =============
    
    def get_markets(self) -> list[MarketInfo]:
        """
        Get information about all markets in this episode.
        
        Returns list of MarketInfo with:
        - ticker
        - best bid/ask for YES and NO
        - status
        - time to close
        """
        self._check_tool_budget()
        
        markets = []
        for ticker in self._tickers:
            book = self._orderbooks.get(ticker)
            
            time_to_close = None
            if self._end_ts and self._current_ts:
                delta = (self._end_ts - self._current_ts).total_seconds()
                time_to_close = max(0, delta)
            
            if book:
                markets.append(MarketInfo(
                    ticker=ticker,
                    event_slug=self._event_slug,
                    yes_best_bid=book.yes_best_bid,
                    yes_best_ask=book.yes_best_ask,
                    no_best_bid=book.no_best_bid,
                    no_best_ask=book.no_best_ask,
                    status="active",
                    time_to_close_seconds=time_to_close,
                ))
            else:
                markets.append(MarketInfo(
                    ticker=ticker,
                    event_slug=self._event_slug,
                    yes_best_bid=None,
                    yes_best_ask=None,
                    no_best_bid=None,
                    no_best_ask=None,
                    status="no_data",
                    time_to_close_seconds=time_to_close,
                ))
        
        return markets
    
    def get_orderbook(
        self,
        ticker: str,
        depth: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Get orderbook for a specific ticker.
        
        Args:
            ticker: Market ticker
            depth: Number of levels to return (default: observation_depth)
            
        Returns:
            Dictionary with:
            - ticker
            - ts
            - yes_bids: [(price_cents, size), ...]
            - no_bids: [(price_cents, size), ...]
            - yes_asks: [(price_cents, size), ...] (derived)
            - no_asks: [(price_cents, size), ...] (derived)
        """
        self._check_tool_budget()
        
        if ticker not in self._orderbooks:
            return None
        
        book = self._orderbooks[ticker]
        depth = depth or self._observation_depth
        
        def levels_to_list(levels: list[OrderbookLevel], max_depth: int) -> list[tuple[int, int]]:
            return [(l.price_cents, l.size) for l in levels[:max_depth]]
        
        return {
            "ticker": ticker,
            "ts": book.ts.isoformat(),
            "sequence_id": book.sequence_id,
            "yes_bids": levels_to_list(book.yes_bids, depth),
            "no_bids": levels_to_list(book.no_bids, depth),
            "yes_asks": levels_to_list(book.get_yes_asks(), depth),
            "no_asks": levels_to_list(book.get_no_asks(), depth),
        }
    
    def place_order(self, order: Order) -> dict:
        """
        Place an order.
        
        Args:
            order: Order to place
            
        Returns:
            Dictionary with order result:
            - filled_count
            - average_price (if filled)
            - total_cost_cents
            - total_fees_cents
            - rejected
            - reject_reason (if rejected)
        """
        self._check_tool_budget()
        
        result = self._place_order(order)
        
        return {
            "filled_count": result.filled_count,
            "average_price": result.average_fill_price,
            "total_cost_cents": result.total_cost_cents,
            "total_fees_cents": result.total_fees_cents,
            "rejected": result.rejected,
            "reject_reason": result.reject_reason,
        }
    
    def get_positions(self) -> dict[str, dict]:
        """
        Get all current positions.
        
        Returns dict mapping ticker to position info:
        - yes_contracts
        - no_contracts
        - yes_avg_cost
        - no_avg_cost
        - realized_pnl_cents
        """
        self._check_tool_budget()
        
        result = {}
        for ticker, pos in self._positions.items():
            if pos.yes_contracts != 0 or pos.no_contracts != 0:
                # Calculate unrealized PnL
                unrealized = 0
                if ticker in self._orderbooks:
                    book = self._orderbooks[ticker]
                    if pos.yes_contracts > 0 and book.yes_best_bid:
                        unrealized += (book.yes_best_bid - pos.yes_avg_cost_cents) * pos.yes_contracts
                    elif pos.yes_contracts < 0 and book.yes_best_ask:
                        unrealized += (pos.yes_avg_cost_cents - book.yes_best_ask) * abs(pos.yes_contracts)
                    if pos.no_contracts > 0 and book.no_best_bid:
                        unrealized += (book.no_best_bid - pos.no_avg_cost_cents) * pos.no_contracts
                    elif pos.no_contracts < 0 and book.no_best_ask:
                        unrealized += (pos.no_avg_cost_cents - book.no_best_ask) * abs(pos.no_contracts)
                
                result[ticker] = {
                    "yes_contracts": pos.yes_contracts,
                    "no_contracts": pos.no_contracts,
                    "yes_avg_cost": pos.yes_avg_cost_cents,
                    "no_avg_cost": pos.no_avg_cost_cents,
                    "realized_pnl_cents": pos.realized_pnl_cents,
                    "unrealized_pnl_cents": int(unrealized),
                }
        
        return result
    
    def get_cash(self) -> dict:
        """
        Get cash and equity information.
        
        Returns:
        - cash_cents
        - equity_cents
        """
        self._check_tool_budget()
        
        return {
            "cash_cents": self._cash_cents,
            "equity_cents": self._equity_cents,
        }
    
    @property
    def current_time(self):
        """Current simulation timestamp."""
        return self._current_ts


class Agent(ABC):
    """
    Abstract base class for trading agents.
    
    Subclass this and implement the `act` method to create a trading agent.
    """
    
    @abstractmethod
    def act(self, ctx: AgentContext) -> None:
        """
        Called at each agent step to make trading decisions.
        
        Use the context to:
        - Query market state: ctx.get_markets(), ctx.get_orderbook(ticker)
        - Check positions: ctx.get_positions(), ctx.get_cash()
        - Place orders: ctx.place_order(order)
        
        Args:
            ctx: AgentContext with tools for market interaction
        """
        pass
    
    def on_episode_start(self, metadata: dict) -> None:
        """
        Called at the start of each episode.
        
        Override to initialize episode-specific state.
        
        Args:
            metadata: Episode metadata dictionary
        """
        pass
    
    def on_episode_end(self, result: dict) -> None:
        """
        Called at the end of each episode.
        
        Override to process results or clean up.
        
        Args:
            result: Episode result dictionary
        """
        pass
