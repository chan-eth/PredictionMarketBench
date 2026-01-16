"""
Portfolio management and accounting.

Tracks positions, cash, and computes mark-to-market values.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .types import (
    Side,
    Action,
    Position,
    OrderResult,
    OrderbookSnapshot,
    EquitySnapshot,
    SettlementResult,
)


@dataclass
class Portfolio:
    """
    Portfolio manager for tracking positions and cash.
    
    Uses liquidation-based mark-to-market (conservative):
    - Long positions valued at best bid
    - Short positions valued at best ask
    """
    
    cash_cents: int
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl_cents: int = 0
    total_fees_paid_cents: int = 0
    total_contracts_traded: int = 0
    total_notional_cents: int = 0
    
    # For slippage tracking
    total_slippage_cents: int = 0
    total_requested_contracts: int = 0
    total_filled_contracts: int = 0
    
    def get_position(self, ticker: str) -> Position:
        """Get position for a ticker, creating if needed."""
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker)
        return self.positions[ticker]
    
    def apply_order_result(
        self,
        result: OrderResult,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> None:
        """
        Apply an order result to the portfolio.
        
        Updates positions, cash, and tracking metrics.
        """
        if result.rejected or result.filled_count == 0:
            # Track unfilled requests
            self.total_requested_contracts += result.order.count
            return
        
        order = result.order
        pos = self.get_position(order.ticker)
        
        # Track metrics
        self.total_fees_paid_cents += result.total_fees_cents
        self.total_contracts_traded += result.filled_count
        self.total_requested_contracts += order.count
        self.total_filled_contracts += result.filled_count
        
        # Calculate notional
        notional = sum(f.price_cents * f.size for f in result.fills)
        self.total_notional_cents += notional
        
        # Calculate slippage (vs mid at order time)
        if orderbook is not None:
            mid = self._get_mid_price(order.side, orderbook)
            if mid is not None:
                expected_cost = int(mid * result.filled_count)
                if order.action == Action.BUY:
                    actual_cost = notional
                    slippage = actual_cost - expected_cost
                else:
                    actual_proceeds = notional
                    slippage = expected_cost - actual_proceeds
                self.total_slippage_cents += slippage
        
        # Update cash
        self.cash_cents -= result.total_cost_cents
        
        # Update position
        avg_price = result.average_fill_price or 0
        
        if order.side == Side.YES:
            if order.action == Action.BUY:
                # Buying YES contracts
                self._update_long_position(
                    pos, "yes", result.filled_count, avg_price
                )
            else:
                # Selling YES contracts
                self._update_short_position(
                    pos, "yes", result.filled_count, avg_price
                )
        else:  # Side.NO
            if order.action == Action.BUY:
                # Buying NO contracts
                self._update_long_position(
                    pos, "no", result.filled_count, avg_price
                )
            else:
                # Selling NO contracts
                self._update_short_position(
                    pos, "no", result.filled_count, avg_price
                )
    
    def _update_long_position(
        self,
        pos: Position,
        side: str,
        count: int,
        price: float,
    ) -> None:
        """Update position when buying contracts."""
        if side == "yes":
            if pos.yes_contracts >= 0:
                # Adding to long or opening new long
                total_cost = pos.yes_avg_cost_cents * pos.yes_contracts + price * count
                pos.yes_contracts += count
                pos.yes_avg_cost_cents = total_cost / pos.yes_contracts if pos.yes_contracts > 0 else 0
            else:
                # Closing short position
                close_count = min(count, -pos.yes_contracts)
                pnl = (pos.yes_avg_cost_cents - price) * close_count
                pos.realized_pnl_cents += int(pnl)
                self.realized_pnl_cents += int(pnl)
                pos.yes_contracts += count
                if pos.yes_contracts > 0:
                    pos.yes_avg_cost_cents = price
        else:  # no
            if pos.no_contracts >= 0:
                total_cost = pos.no_avg_cost_cents * pos.no_contracts + price * count
                pos.no_contracts += count
                pos.no_avg_cost_cents = total_cost / pos.no_contracts if pos.no_contracts > 0 else 0
            else:
                close_count = min(count, -pos.no_contracts)
                pnl = (pos.no_avg_cost_cents - price) * close_count
                pos.realized_pnl_cents += int(pnl)
                self.realized_pnl_cents += int(pnl)
                pos.no_contracts += count
                if pos.no_contracts > 0:
                    pos.no_avg_cost_cents = price
    
    def _update_short_position(
        self,
        pos: Position,
        side: str,
        count: int,
        price: float,
    ) -> None:
        """Update position when selling contracts."""
        if side == "yes":
            if pos.yes_contracts <= 0:
                # Adding to short or opening new short
                total_cost = pos.yes_avg_cost_cents * abs(pos.yes_contracts) + price * count
                pos.yes_contracts -= count
                pos.yes_avg_cost_cents = total_cost / abs(pos.yes_contracts) if pos.yes_contracts != 0 else 0
            else:
                # Closing long position
                close_count = min(count, pos.yes_contracts)
                pnl = (price - pos.yes_avg_cost_cents) * close_count
                pos.realized_pnl_cents += int(pnl)
                self.realized_pnl_cents += int(pnl)
                pos.yes_contracts -= count
                if pos.yes_contracts < 0:
                    pos.yes_avg_cost_cents = price
        else:  # no
            if pos.no_contracts <= 0:
                total_cost = pos.no_avg_cost_cents * abs(pos.no_contracts) + price * count
                pos.no_contracts -= count
                pos.no_avg_cost_cents = total_cost / abs(pos.no_contracts) if pos.no_contracts != 0 else 0
            else:
                close_count = min(count, pos.no_contracts)
                pnl = (price - pos.no_avg_cost_cents) * close_count
                pos.realized_pnl_cents += int(pnl)
                self.realized_pnl_cents += int(pnl)
                pos.no_contracts -= count
                if pos.no_contracts < 0:
                    pos.no_avg_cost_cents = price
    
    def settle_position(
        self,
        ticker: str,
        result: SettlementResult,
    ) -> int:
        """
        Settle a position based on market outcome.
        
        YES contracts pay 100 cents if YES, 0 if NO.
        NO contracts pay 100 cents if NO, 0 if YES.
        
        Returns settlement proceeds in cents.
        """
        if ticker not in self.positions:
            return 0
        
        pos = self.positions[ticker]
        proceeds = 0
        
        is_yes = result.result == "YES"
        
        # Settle YES contracts
        if pos.yes_contracts != 0:
            if pos.yes_contracts > 0:
                # Long YES
                if is_yes:
                    proceeds += pos.yes_contracts * 100
                # else: contracts expire worthless
                pnl = (100 if is_yes else 0) - pos.yes_avg_cost_cents
                pos.realized_pnl_cents += int(pnl * pos.yes_contracts)
            else:
                # Short YES (negative position)
                if is_yes:
                    proceeds += pos.yes_contracts * 100  # Negative, so pay out
                pnl = pos.yes_avg_cost_cents - (100 if is_yes else 0)
                pos.realized_pnl_cents += int(pnl * abs(pos.yes_contracts))
            pos.yes_contracts = 0
        
        # Settle NO contracts
        if pos.no_contracts != 0:
            if pos.no_contracts > 0:
                # Long NO
                if not is_yes:
                    proceeds += pos.no_contracts * 100
                pnl = (100 if not is_yes else 0) - pos.no_avg_cost_cents
                pos.realized_pnl_cents += int(pnl * pos.no_contracts)
            else:
                # Short NO
                if not is_yes:
                    proceeds += pos.no_contracts * 100  # Negative
                pnl = pos.no_avg_cost_cents - (100 if not is_yes else 0)
                pos.realized_pnl_cents += int(pnl * abs(pos.no_contracts))
            pos.no_contracts = 0
        
        self.cash_cents += proceeds
        self.realized_pnl_cents = sum(p.realized_pnl_cents for p in self.positions.values())
        
        return proceeds
    
    def _get_mid_price(
        self,
        side: Side,
        orderbook: OrderbookSnapshot,
    ) -> Optional[float]:
        """Get mid price for a side."""
        if side == Side.YES:
            return orderbook.yes_mid
        else:
            return orderbook.no_mid
    
    def get_position_value(
        self,
        orderbooks: dict[str, OrderbookSnapshot],
    ) -> int:
        """
        Calculate total position value using liquidation pricing.
        
        Longs valued at best bid, shorts at best ask.
        """
        total = 0
        
        for ticker, pos in self.positions.items():
            if ticker not in orderbooks:
                # No orderbook - use 50 cents as fallback
                if pos.yes_contracts != 0:
                    total += pos.yes_contracts * 50
                if pos.no_contracts != 0:
                    total += pos.no_contracts * 50
                continue
            
            book = orderbooks[ticker]
            
            # Value YES position
            if pos.yes_contracts > 0:
                # Long: value at best bid (what we could sell for)
                bid = book.yes_best_bid
                total += pos.yes_contracts * (bid if bid else 50)
            elif pos.yes_contracts < 0:
                # Short: value at best ask (what we'd pay to cover)
                ask = book.yes_best_ask
                total += pos.yes_contracts * (ask if ask else 50)  # Negative
            
            # Value NO position
            if pos.no_contracts > 0:
                bid = book.no_best_bid
                total += pos.no_contracts * (bid if bid else 50)
            elif pos.no_contracts < 0:
                ask = book.no_best_ask
                total += pos.no_contracts * (ask if ask else 50)
        
        return total
    
    def get_equity(
        self,
        orderbooks: dict[str, OrderbookSnapshot],
    ) -> int:
        """Get total equity (cash + position value)."""
        return self.cash_cents + self.get_position_value(orderbooks)
    
    def get_equity_snapshot(
        self,
        ts: datetime,
        orderbooks: dict[str, OrderbookSnapshot],
    ) -> EquitySnapshot:
        """Create an equity snapshot at current time."""
        position_value = self.get_position_value(orderbooks)
        return EquitySnapshot(
            ts=ts,
            cash_cents=self.cash_cents,
            position_value_cents=position_value,
            equity_cents=self.cash_cents + position_value,
        )
    
    @property
    def fill_ratio(self) -> float:
        """Fraction of requested contracts that were filled."""
        if self.total_requested_contracts == 0:
            return 1.0
        return self.total_filled_contracts / self.total_requested_contracts
