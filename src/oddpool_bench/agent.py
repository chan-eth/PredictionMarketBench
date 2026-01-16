"""
Agent interface for the benchmark.

Defines the abstract base class that trading agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable

from .types import (
    Order,
    Position,
    MarketInfo,
    OrderbookSnapshot,
    OrderbookLevel,
    RestingOrder,
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
        place_order_callback: Callable,
        cancel_order_callback: Optional[Callable] = None,
        observation_depth: int = -1,  # -1 means full depth
        resting_orders: Optional[list[RestingOrder]] = None,
        execution_mode: str = "taker_only",
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
        self._cancel_order = cancel_order_callback
        self._observation_depth = observation_depth
        self._resting_orders = resting_orders or []
        self._execution_mode = execution_mode
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
            depth: Number of levels to return (default: all available)
            
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
        # -1 means full depth
        max_depth = depth if depth and depth > 0 else (
            self._observation_depth if self._observation_depth > 0 else 999999
        )
        
        def levels_to_list(levels: list[OrderbookLevel], max_depth: int) -> list[tuple[int, int]]:
            return [(l.price_cents, l.size) for l in levels[:max_depth]]
        
        return {
            "ticker": ticker,
            "ts": book.ts.isoformat(),
            "sequence_id": book.sequence_id,
            "yes_bids": levels_to_list(book.yes_bids, max_depth),
            "no_bids": levels_to_list(book.no_bids, max_depth),
            "yes_asks": levels_to_list(book.get_yes_asks(), max_depth),
            "no_asks": levels_to_list(book.get_no_asks(), max_depth),
        }
    
    def place_order(self, order: Order) -> dict:
        """
        Place an order.
        
        Args:
            order: Order to place
            
        Returns:
            Dictionary with order result:
            - rejected: bool
            - rejection_reason: str or None
            - fill: dict or None with:
                - count: number of contracts filled
                - avg_price_cents: average fill price
                - total_cost_cents: total cost including fees
                - fee_cents: total fees
            - resting: dict or None if order is resting (maker mode)
        """
        self._check_tool_budget()
        
        result = self._place_order(order)
        
        # Build fill info
        fill_info = None
        if result.filled_count > 0:
            fill_info = {
                "count": result.filled_count,
                "avg_price_cents": result.average_fill_price,
                "total_cost_cents": result.total_cost_cents,
                "fee_cents": result.total_fees_cents,
            }
        
        # Build resting order info
        resting_info = None
        if result.resting_order:
            resting_info = {
                "order_id": result.resting_order.order_id,
                "ticker": result.resting_order.original_order.ticker,
                "side": result.resting_order.contract_side.name,
                "price_cents": result.resting_order.price_cents,
                "remaining_count": result.resting_order.remaining_count,
            }
        
        return {
            "rejected": result.rejected,
            "rejection_reason": result.reject_reason,
            "fill": fill_info,
            "resting": resting_info,
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
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a resting order (maker_taker mode only).
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if order was canceled, False if not found
        """
        self._check_tool_budget()
        
        if self._cancel_order is None:
            return False
        
        return self._cancel_order(order_id)
    
    def get_resting_orders(self, ticker: Optional[str] = None) -> list[dict]:
        """
        Get all resting (maker) orders.
        
        Args:
            ticker: Optional ticker to filter by
            
        Returns:
            List of resting order info:
            - order_id
            - ticker
            - side (YES/NO)
            - action (BUY/SELL)
            - price_cents
            - remaining_count
            - original_count
            - placed_ts
            - fills (list of fill info)
        """
        self._check_tool_budget()
        
        result = []
        for resting in self._resting_orders:
            if ticker and resting.original_order.ticker != ticker:
                continue
            
            result.append({
                "order_id": resting.order_id,
                "ticker": resting.original_order.ticker,
                "side": resting.original_order.side.value,
                "action": resting.original_order.action.value,
                "price_cents": resting.price_cents,
                "remaining_count": resting.remaining_count,
                "original_count": resting.original_order.count,
                "placed_ts": resting.placed_ts.isoformat(),
                "env_ahead": resting.env_ahead,
                "fills": [
                    {
                        "price_cents": f.price_cents,
                        "size": f.size,
                        "fee_cents": f.fee_cents,
                        "fill_ts": f.fill_ts.isoformat() if f.fill_ts else None,
                    }
                    for f in resting.fills
                ],
            })
        
        return result
    
    @property
    def execution_mode(self) -> str:
        """Current execution mode (taker_only or maker_taker)."""
        return self._execution_mode
    
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
