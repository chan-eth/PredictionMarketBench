"""
Maker queue management for resting limit orders.

Implements a queue-position model where:
1. Agent orders track their position relative to environment liquidity
2. Trade prints from historical tape consume liquidity from queues
3. Agent fills occur when env_ahead is consumed and trade volume remains
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

from .types import (
    Side,
    Order,
    Fill,
    FillType,
    RestingOrder,
    OrderbookSnapshot,
)
from .fees import FeeModel


@dataclass
class LevelQueue:
    """
    Queue state for a single price level on one side.
    
    Tracks environment volume ahead and agent orders in FIFO.
    """
    ticker: str
    contract_side: Side  # YES or NO
    price_cents: int
    env_ahead: int = 0  # Non-agent volume ahead of first agent order
    agent_orders: list[RestingOrder] = field(default_factory=list)
    total_consumed: int = 0  # Total volume consumed at this level (debug)
    
    def add_order(self, order: RestingOrder) -> None:
        """Add a new agent order to the queue (FIFO)."""
        self.agent_orders.append(order)
    
    def remove_order(self, order_id: str) -> Optional[RestingOrder]:
        """Remove and return an order by ID, or None if not found."""
        for i, order in enumerate(self.agent_orders):
            if order.order_id == order_id:
                return self.agent_orders.pop(i)
        return None
    
    def consume_volume(
        self,
        volume: int,
        ts: datetime,
        fee_model: FeeModel,
    ) -> list[tuple[RestingOrder, Fill]]:
        """
        Consume volume from this level due to a trade print.
        
        Returns list of (order, fill) pairs for agent fills.
        """
        fills = []
        remaining = volume
        self.total_consumed += volume
        
        # First consume from env_ahead
        if self.env_ahead > 0:
            consumed = min(self.env_ahead, remaining)
            self.env_ahead -= consumed
            remaining -= consumed
        
        # Then fill agent orders FIFO
        while remaining > 0 and self.agent_orders:
            order = self.agent_orders[0]
            fill_size = min(order.remaining_count, remaining)
            
            # Calculate maker fee
            fee = fee_model.calculate_maker_fee(self.price_cents, fill_size)
            
            fill = Fill(
                price_cents=self.price_cents,
                size=fill_size,
                fee_cents=fee,
                fill_type=FillType.MAKER,
                fill_ts=ts,
            )
            
            order.remaining_count -= fill_size
            order.fills.append(fill)
            remaining -= fill_size
            
            fills.append((order, fill))
            
            # Remove fully filled orders
            if order.remaining_count <= 0:
                self.agent_orders.pop(0)
        
        return fills
    
    @property
    def total_agent_size(self) -> int:
        """Total remaining size of agent orders at this level."""
        return sum(o.remaining_count for o in self.agent_orders)
    
    @property
    def is_empty(self) -> bool:
        """True if no agent orders at this level."""
        return len(self.agent_orders) == 0


class MakerQueueManager:
    """
    Manages all maker order queues for an episode.
    
    Key concepts:
    - Each (ticker, contract_side, price_cents) has its own LevelQueue
    - New orders get queue position based on current book + existing agent orders
    - Trade prints consume queue volume and may fill agent orders
    """
    
    def __init__(self, fee_model: FeeModel):
        self.fee_model = fee_model
        # Key: (ticker, contract_side, price_cents)
        self.queues: dict[tuple[str, Side, int], LevelQueue] = {}
        # Track all resting orders by ID
        self.orders_by_id: dict[str, RestingOrder] = {}
        # Order ID counter
        self._order_counter = 0
    
    def _get_or_create_queue(
        self,
        ticker: str,
        contract_side: Side,
        price_cents: int,
    ) -> LevelQueue:
        """Get or create a queue for the given level."""
        key = (ticker, contract_side, price_cents)
        if key not in self.queues:
            self.queues[key] = LevelQueue(
                ticker=ticker,
                contract_side=contract_side,
                price_cents=price_cents,
            )
        return self.queues[key]
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self._order_counter += 1
        return f"agent-{self._order_counter}"
    
    def place_order(
        self,
        order: Order,
        current_book: OrderbookSnapshot,
        ts: datetime,
    ) -> RestingOrder:
        """
        Place a resting order in the queue.
        
        Args:
            order: The limit order to place
            current_book: Current orderbook state for queue position
            ts: Placement timestamp
            
        Returns:
            RestingOrder tracking the resting position
        """
        # Convert to canonical bid
        contract_side, price_cents = order.to_canonical_bid()
        
        # Get current book size at this level
        if contract_side == Side.YES:
            book_levels = current_book.yes_bids
        else:
            book_levels = current_book.no_bids
        
        # Find size at this price level
        env_size = 0
        for level in book_levels:
            if level.price_cents == price_cents:
                env_size = level.size
                break
        
        # Get or create queue
        queue = self._get_or_create_queue(order.ticker, contract_side, price_cents)
        
        # If first agent order at this level, set env_ahead
        if queue.is_empty:
            queue.env_ahead = env_size
        
        # Agent's existing orders at this level are also ahead
        agent_ahead = queue.total_agent_size
        
        # Create resting order
        order_id = order.order_id or self._generate_order_id()
        resting = RestingOrder(
            order_id=order_id,
            original_order=order,
            contract_side=contract_side,
            price_cents=price_cents,
            remaining_count=order.count,
            placed_ts=ts,
            env_ahead=env_size + agent_ahead,  # Total ahead at placement
        )
        
        # Add to queue
        queue.add_order(resting)
        self.orders_by_id[order_id] = resting
        
        return resting
    
    def cancel_order(self, order_id: str) -> Optional[RestingOrder]:
        """
        Cancel a resting order.
        
        Returns the canceled order or None if not found.
        """
        if order_id not in self.orders_by_id:
            return None
        
        order = self.orders_by_id[order_id]
        key = (order.original_order.ticker, order.contract_side, order.price_cents)
        
        if key in self.queues:
            self.queues[key].remove_order(order_id)
        
        del self.orders_by_id[order_id]
        return order
    
    def process_trade(
        self,
        ticker: str,
        taker_side: Side,
        trade_price_cents: int,
        volume: int,
        ts: datetime,
    ) -> list[tuple[RestingOrder, Fill]]:
        """
        Process a trade print and return any agent fills.
        
        Trade mapping:
        - Taker bought YES at p → consumed NO bids at (100-p)
        - Taker bought NO at p → consumed YES bids at (100-p)
        
        Args:
            ticker: Market ticker
            taker_side: Direction of taker (YES = bought YES, NO = bought NO)
            trade_price_cents: Price of the trade
            volume: Number of contracts traded
            ts: Trade timestamp
            
        Returns:
            List of (order, fill) pairs for agent fills
        """
        # Determine passive side and price
        passive_side = Side.NO if taker_side == Side.YES else Side.YES
        passive_price = 100 - trade_price_cents
        
        # Get queue for this level
        key = (ticker, passive_side, passive_price)
        if key not in self.queues:
            return []
        
        queue = self.queues[key]
        fills = queue.consume_volume(volume, ts, self.fee_model)
        
        # Clean up empty queues
        if queue.is_empty and queue.env_ahead == 0:
            del self.queues[key]
        
        return fills
    
    def get_resting_orders(self, ticker: Optional[str] = None) -> list[RestingOrder]:
        """Get all resting orders, optionally filtered by ticker."""
        if ticker is None:
            return list(self.orders_by_id.values())
        return [o for o in self.orders_by_id.values() if o.original_order.ticker == ticker]
    
    def get_order(self, order_id: str) -> Optional[RestingOrder]:
        """Get a resting order by ID."""
        return self.orders_by_id.get(order_id)
    
    def update_env_from_snapshot(
        self,
        snapshot: OrderbookSnapshot,
        mode: str = "trade_only",
    ) -> None:
        """
        Optionally update env_ahead based on new orderbook snapshot.
        
        mode="trade_only": Don't update env_ahead (conservative)
        mode="reconciled": Infer cancellations and adjust env_ahead
        """
        if mode == "trade_only":
            # In trade_only mode, we don't adjust env_ahead based on snapshots
            # Agent position only improves via trade consumption
            return
        
        # Mode: reconciled - infer cancellations
        # For each queue at this ticker, compare with snapshot
        for (ticker, side, price), queue in self.queues.items():
            if ticker != snapshot.ticker:
                continue
            
            # Get current book size at this level
            if side == Side.YES:
                book_levels = snapshot.yes_bids
            else:
                book_levels = snapshot.no_bids
            
            current_size = 0
            for level in book_levels:
                if level.price_cents == price:
                    current_size = level.size
                    break
            
            # If current size is less than env_ahead, reduce env_ahead
            # (This assumes some env orders were canceled)
            if current_size < queue.env_ahead:
                queue.env_ahead = current_size
