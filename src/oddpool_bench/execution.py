"""
Execution engine for order processing.

Handles order matching, book walking, and fill generation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .types import (
    Side,
    Action,
    OrderType,
    Order,
    OrderResult,
    Fill,
    OrderbookSnapshot,
    OrderbookLevel,
)
from .fees import FeeModel


@dataclass
class ExecutionEngine:
    """
    Execution engine for processing orders against orderbook.
    
    In v0 (taker-only mode), only orders that execute immediately are allowed.
    Market orders always execute as taker.
    Limit orders must be "crossing" (price >= ask for buys, price <= bid for sells).
    """
    
    fee_model: FeeModel
    execution_mode: str = "taker_only"
    
    def execute_order(
        self,
        order: Order,
        orderbook: OrderbookSnapshot,
        ts: datetime,
    ) -> OrderResult:
        """
        Execute an order against the current orderbook.
        
        Args:
            order: The order to execute
            orderbook: Current orderbook state for the ticker
            ts: Simulation timestamp
            
        Returns:
            OrderResult with fill information
        """
        if order.ticker != orderbook.ticker:
            return OrderResult(
                order=order,
                ts=ts,
                filled_count=0,
                fills=[],
                total_cost_cents=0,
                total_fees_cents=0,
                rejected=True,
                reject_reason=f"Ticker mismatch: order={order.ticker}, book={orderbook.ticker}",
            )
        
        # Get the executable book (asks we can hit for buys, bids for sells)
        executable_levels = self._get_executable_levels(order, orderbook)
        
        if not executable_levels:
            # No liquidity
            return OrderResult(
                order=order,
                ts=ts,
                filled_count=0,
                fills=[],
                total_cost_cents=0,
                total_fees_cents=0,
                rejected=False,  # Not rejected, just no fill
            )
        
        # For limit orders in taker-only mode, check if order crosses
        if order.order_type == OrderType.LIMIT:
            best_price = executable_levels[0].price_cents
            is_buy = order.action == Action.BUY
            
            if is_buy and order.limit_price_cents < best_price:
                # Limit price below best ask - would rest as maker (not allowed in v0)
                return OrderResult(
                    order=order,
                    ts=ts,
                    filled_count=0,
                    fills=[],
                    total_cost_cents=0,
                    total_fees_cents=0,
                    rejected=True,
                    reject_reason="Limit order does not cross (taker-only mode)",
                )
            elif not is_buy and order.limit_price_cents > best_price:
                # Limit price above best bid - would rest as maker
                return OrderResult(
                    order=order,
                    ts=ts,
                    filled_count=0,
                    fills=[],
                    total_cost_cents=0,
                    total_fees_cents=0,
                    rejected=True,
                    reject_reason="Limit order does not cross (taker-only mode)",
                )
        
        # Walk the book and fill
        fills = []
        remaining = order.count
        total_cost = 0
        total_fees = 0
        
        for level in executable_levels:
            if remaining <= 0:
                break
            
            # For limit orders, check price constraint
            if order.order_type == OrderType.LIMIT:
                is_buy = order.action == Action.BUY
                if is_buy and level.price_cents > order.limit_price_cents:
                    break  # Price too high
                elif not is_buy and level.price_cents < order.limit_price_cents:
                    break  # Price too low
            
            fill_size = min(remaining, level.size)
            fill_price = level.price_cents
            
            # Calculate fee for this fill
            fee = self.fee_model.calculate_taker_fee(fill_price, fill_size)
            
            fills.append(Fill(
                price_cents=fill_price,
                size=fill_size,
                fee_cents=fee,
            ))
            
            # Update totals
            if order.action == Action.BUY:
                # Buying: pay price + fee
                total_cost += fill_price * fill_size + fee
            else:
                # Selling: receive price - fee
                total_cost -= fill_price * fill_size - fee
            
            total_fees += fee
            remaining -= fill_size
        
        filled_count = order.count - remaining
        
        return OrderResult(
            order=order,
            ts=ts,
            filled_count=filled_count,
            fills=fills,
            total_cost_cents=total_cost,
            total_fees_cents=total_fees,
        )
    
    def _get_executable_levels(
        self,
        order: Order,
        orderbook: OrderbookSnapshot,
    ) -> list[OrderbookLevel]:
        """
        Get the price levels that can fill this order.
        
        For buys: we consume asks (derived from opposite side bids)
        For sells: we consume bids
        
        Returns levels sorted best-first for execution.
        """
        is_buy = order.action == Action.BUY
        
        if order.side == Side.YES:
            if is_buy:
                # Buying YES = consuming YES asks (derived from NO bids)
                return orderbook.get_yes_asks()
            else:
                # Selling YES = consuming YES bids
                # Sort by price descending (best bid = highest price first)
                return sorted(orderbook.yes_bids, key=lambda x: -x.price_cents)
        else:  # Side.NO
            if is_buy:
                # Buying NO = consuming NO asks (derived from YES bids)
                return orderbook.get_no_asks()
            else:
                # Selling NO = consuming NO bids
                # Sort by price descending (best bid = highest price first)
                return sorted(orderbook.no_bids, key=lambda x: -x.price_cents)
    
    def estimate_fill(
        self,
        order: Order,
        orderbook: OrderbookSnapshot,
    ) -> tuple[int, int]:
        """
        Estimate fill quantity and cost without executing.
        
        Returns:
            Tuple of (estimated_fill_count, estimated_cost_cents)
        """
        # Use same logic as execute_order but don't create full result
        executable_levels = self._get_executable_levels(order, orderbook)
        
        if not executable_levels:
            return (0, 0)
        
        remaining = order.count
        total_cost = 0
        
        for level in executable_levels:
            if remaining <= 0:
                break
            
            if order.order_type == OrderType.LIMIT:
                is_buy = order.action == Action.BUY
                if is_buy and level.price_cents > order.limit_price_cents:
                    break
                elif not is_buy and level.price_cents < order.limit_price_cents:
                    break
            
            fill_size = min(remaining, level.size)
            fill_price = level.price_cents
            fee = self.fee_model.calculate_taker_fee(fill_price, fill_size)
            
            if order.action == Action.BUY:
                total_cost += fill_price * fill_size + fee
            else:
                total_cost -= fill_price * fill_size - fee
            
            remaining -= fill_size
        
        filled_count = order.count - remaining
        return (filled_count, total_cost)
