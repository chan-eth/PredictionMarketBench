"""
Execution engine for order processing.

Handles order matching, book walking, and fill generation.
Supports both taker-only (v0) and maker+taker (v1) modes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .types import (
    Side,
    Action,
    OrderType,
    TimeInForce,
    Order,
    OrderResult,
    Fill,
    FillType,
    RestingOrder,
    OrderbookSnapshot,
    OrderbookLevel,
)
from .fees import FeeModel


@dataclass
class ExecutionEngine:
    """
    Execution engine for processing orders against orderbook.
    
    Modes:
    - taker_only: Orders must cross immediately or are rejected (v0)
    - maker_taker: Orders can rest as maker if they don't cross (v1)
    """
    
    fee_model: FeeModel
    execution_mode: str = "taker_only"  # "taker_only" or "maker_taker"
    
    def execute_order(
        self,
        order: Order,
        orderbook: OrderbookSnapshot,
        ts: datetime,
    ) -> OrderResult:
        """
        Execute an order against the current orderbook.
        
        For taker fills (crossing orders), fills happen immediately.
        For maker orders (non-crossing GTC), returns with resting_order set.
        
        Args:
            order: The order to execute
            orderbook: Current orderbook state for the ticker
            ts: Simulation timestamp
            
        Returns:
            OrderResult with fill information (and resting_order for GTC)
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
        
        # Check if order would cross
        would_cross = False
        if executable_levels and order.order_type == OrderType.LIMIT:
            best_price = executable_levels[0].price_cents
            is_buy = order.action == Action.BUY
            if is_buy:
                would_cross = order.limit_price_cents >= best_price
            else:
                would_cross = order.limit_price_cents <= best_price
        elif order.order_type == OrderType.MARKET:
            would_cross = bool(executable_levels)
        
        # Handle POST_ONLY: reject if would cross
        if order.time_in_force == TimeInForce.POST_ONLY:
            if would_cross:
                return OrderResult(
                    order=order,
                    ts=ts,
                    filled_count=0,
                    fills=[],
                    total_cost_cents=0,
                    total_fees_cents=0,
                    rejected=True,
                    reject_reason="POST_ONLY order would cross",
                )
            # Will rest as maker (handled below)
        
        # Handle non-crossing orders based on mode and TIF
        if not would_cross:
            if order.order_type == OrderType.MARKET:
                # Market orders with no liquidity
                return OrderResult(
                    order=order,
                    ts=ts,
                    filled_count=0,
                    fills=[],
                    total_cost_cents=0,
                    total_fees_cents=0,
                    rejected=False,
                )
            
            # Limit order doesn't cross
            if self.execution_mode == "taker_only":
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
            
            # maker_taker mode: check TIF
            if order.time_in_force == TimeInForce.IOC:
                # IOC cancels if doesn't cross
                return OrderResult(
                    order=order,
                    ts=ts,
                    filled_count=0,
                    fills=[],
                    total_cost_cents=0,
                    total_fees_cents=0,
                    rejected=False,  # Not rejected, just no fill
                )
            
            # GTC or POST_ONLY: rest as maker
            # Return result indicating order should rest
            # (actual resting handled by caller/simulator)
            return OrderResult(
                order=order,
                ts=ts,
                filled_count=0,
                fills=[],
                total_cost_cents=0,
                total_fees_cents=0,
                rejected=False,
                resting_order=None,  # Caller will create RestingOrder
            )
        
        # Execute crossing portion (taker fills)
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
            
            # Calculate taker fee
            fee = self.fee_model.calculate_taker_fee(fill_price, fill_size)
            
            fills.append(Fill(
                price_cents=fill_price,
                size=fill_size,
                fee_cents=fee,
                fill_type=FillType.TAKER,
                fill_ts=ts,
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
        
        # Check if remaining should rest (GTC in maker_taker mode)
        should_rest = (
            remaining > 0 and
            self.execution_mode == "maker_taker" and
            order.time_in_force == TimeInForce.GTC and
            order.order_type == OrderType.LIMIT
        )
        
        return OrderResult(
            order=order,
            ts=ts,
            filled_count=filled_count,
            fills=fills,
            total_cost_cents=total_cost,
            total_fees_cents=total_fees,
            # resting_order will be set by caller if should_rest
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
