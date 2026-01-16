"""Tests for portfolio accounting."""

import pytest
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oddpool_bench.types import (
    Side,
    Action,
    OrderType,
    Order,
    OrderResult,
    Fill,
    OrderbookLevel,
    OrderbookSnapshot,
    SettlementResult,
)
from oddpool_bench.portfolio import Portfolio


class TestPortfolio:
    """Tests for portfolio accounting."""
    
    @pytest.fixture
    def portfolio(self):
        """Create a portfolio with $100 initial cash."""
        return Portfolio(cash_cents=10000)
    
    @pytest.fixture
    def sample_book(self):
        """Create a sample orderbook."""
        return OrderbookSnapshot(
            ts=datetime.now(),
            sequence_id=1,
            ticker="TEST",
            yes_bids=[OrderbookLevel(price_cents=45, size=100)],
            no_bids=[OrderbookLevel(price_cents=52, size=100)],
        )
    
    def test_buy_updates_position_and_cash(self, portfolio):
        """Buying contracts updates position and deducts cash."""
        order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.BUY,
            order_type=OrderType.MARKET,
            count=10,
        )
        
        result = OrderResult(
            order=order,
            ts=datetime.now(),
            filled_count=10,
            fills=[Fill(price_cents=50, size=10, fee_cents=1)],
            total_cost_cents=501,  # 50*10 + 1 fee
            total_fees_cents=1,
        )
        
        portfolio.apply_order_result(result)
        
        pos = portfolio.get_position("TEST")
        assert pos.yes_contracts == 10
        assert portfolio.cash_cents == 10000 - 501
    
    def test_sell_updates_position_and_adds_cash(self, portfolio):
        """Selling contracts updates position and adds cash (minus fees)."""
        # First buy some
        buy_order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.BUY,
            order_type=OrderType.MARKET,
            count=10,
        )
        buy_result = OrderResult(
            order=buy_order,
            ts=datetime.now(),
            filled_count=10,
            fills=[Fill(price_cents=50, size=10, fee_cents=1)],
            total_cost_cents=501,
            total_fees_cents=1,
        )
        portfolio.apply_order_result(buy_result)
        
        # Then sell
        sell_order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.SELL,
            order_type=OrderType.MARKET,
            count=5,
        )
        sell_result = OrderResult(
            order=sell_order,
            ts=datetime.now(),
            filled_count=5,
            fills=[Fill(price_cents=55, size=5, fee_cents=1)],
            total_cost_cents=-(55*5 - 1),  # Proceeds minus fee
            total_fees_cents=1,
        )
        portfolio.apply_order_result(sell_result)
        
        pos = portfolio.get_position("TEST")
        assert pos.yes_contracts == 5
    
    def test_settlement_yes_wins(self, portfolio):
        """Settlement with YES outcome pays 100 cents per YES contract."""
        # Set up a YES position
        portfolio.positions["TEST"] = portfolio.get_position("TEST")
        portfolio.positions["TEST"].yes_contracts = 10
        portfolio.positions["TEST"].yes_avg_cost_cents = 40.0
        
        settlement = SettlementResult(
            ticker="TEST",
            result="YES",
            settled_ts=datetime.now(),
        )
        
        initial_cash = portfolio.cash_cents
        proceeds = portfolio.settle_position("TEST", settlement)
        
        # YES contracts pay 100 each when YES wins
        assert proceeds == 1000  # 10 * 100
        assert portfolio.cash_cents == initial_cash + 1000
        assert portfolio.positions["TEST"].yes_contracts == 0
    
    def test_settlement_no_wins(self, portfolio):
        """Settlement with NO outcome pays 0 for YES contracts."""
        portfolio.positions["TEST"] = portfolio.get_position("TEST")
        portfolio.positions["TEST"].yes_contracts = 10
        portfolio.positions["TEST"].yes_avg_cost_cents = 40.0
        
        settlement = SettlementResult(
            ticker="TEST",
            result="NO",
            settled_ts=datetime.now(),
        )
        
        initial_cash = portfolio.cash_cents
        proceeds = portfolio.settle_position("TEST", settlement)
        
        # YES contracts pay 0 when NO wins
        assert proceeds == 0
        assert portfolio.cash_cents == initial_cash
        assert portfolio.positions["TEST"].yes_contracts == 0
    
    def test_position_value_liquidation_pricing(self, portfolio, sample_book):
        """Position value uses liquidation pricing."""
        portfolio.positions["TEST"] = portfolio.get_position("TEST")
        portfolio.positions["TEST"].yes_contracts = 10
        
        # Long YES valued at best YES bid (45)
        value = portfolio.get_position_value({"TEST": sample_book})
        assert value == 450  # 10 * 45
    
    def test_fill_ratio_tracking(self, portfolio):
        """Fill ratio tracks filled vs requested contracts."""
        # Partial fill
        order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.BUY,
            order_type=OrderType.MARKET,
            count=100,
        )
        result = OrderResult(
            order=order,
            ts=datetime.now(),
            filled_count=80,  # Only 80 of 100 filled
            fills=[Fill(price_cents=50, size=80, fee_cents=1)],
            total_cost_cents=4001,
            total_fees_cents=1,
        )
        portfolio.apply_order_result(result)
        
        assert portfolio.total_requested_contracts == 100
        assert portfolio.total_filled_contracts == 80
        assert portfolio.fill_ratio == 0.8
