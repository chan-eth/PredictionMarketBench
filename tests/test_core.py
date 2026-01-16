"""Tests for core benchmark components."""

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
    OrderbookLevel,
    OrderbookSnapshot,
)
from oddpool_bench.fees import KalshiOct2025FeeModel
from oddpool_bench.execution import ExecutionEngine


class TestOrderbookSnapshot:
    """Tests for orderbook operations."""
    
    def test_derive_yes_asks_from_no_bids(self):
        """YES asks are derived from NO bids with price flip."""
        book = OrderbookSnapshot(
            ts=datetime.now(),
            sequence_id=1,
            ticker="TEST",
            yes_bids=[],
            no_bids=[
                OrderbookLevel(price_cents=95, size=100),  # NO bid at 95
                OrderbookLevel(price_cents=90, size=50),   # NO bid at 90
            ],
        )
        
        yes_asks = book.get_yes_asks()
        
        # YES ask = 100 - NO bid
        # So NO bid 95 -> YES ask 5
        # NO bid 90 -> YES ask 10
        assert len(yes_asks) == 2
        assert yes_asks[0].price_cents == 5   # Best ask (lowest) first
        assert yes_asks[0].size == 100
        assert yes_asks[1].price_cents == 10
        assert yes_asks[1].size == 50
    
    def test_derive_no_asks_from_yes_bids(self):
        """NO asks are derived from YES bids with price flip."""
        book = OrderbookSnapshot(
            ts=datetime.now(),
            sequence_id=1,
            ticker="TEST",
            yes_bids=[
                OrderbookLevel(price_cents=40, size=200),
                OrderbookLevel(price_cents=35, size=100),
            ],
            no_bids=[],
        )
        
        no_asks = book.get_no_asks()
        
        # NO ask = 100 - YES bid
        # YES bid 40 -> NO ask 60
        # YES bid 35 -> NO ask 65
        assert len(no_asks) == 2
        assert no_asks[0].price_cents == 60  # Best ask first
        assert no_asks[1].price_cents == 65
    
    def test_best_bid_ask_properties(self):
        """Test convenience properties for best prices."""
        book = OrderbookSnapshot(
            ts=datetime.now(),
            sequence_id=1,
            ticker="TEST",
            yes_bids=[
                OrderbookLevel(price_cents=45, size=10),
                OrderbookLevel(price_cents=40, size=20),
            ],
            no_bids=[
                OrderbookLevel(price_cents=52, size=15),
                OrderbookLevel(price_cents=50, size=25),
            ],
        )
        
        assert book.yes_best_bid == 45
        assert book.yes_best_ask == 48  # 100 - 52 (best NO bid)
        assert book.no_best_bid == 52
        assert book.no_best_ask == 55  # 100 - 45 (best YES bid)


class TestKalshiFeeModel:
    """Tests for the Kalshi fee model."""
    
    def test_taker_fee_at_50(self):
        """Taker fee at 50 cents (maximum potential payout)."""
        fee_model = KalshiOct2025FeeModel()
        
        # At 50 cents, potential payout = min(50, 50) = 50
        # Fee = ceil(50 * 1 * 0.02) = ceil(1.0) = 1
        fee = fee_model.calculate_taker_fee(price_cents=50, count=1)
        assert fee == 1
        
        # 10 contracts
        # Fee = ceil(50 * 10 * 0.02) = ceil(10) = 10
        fee = fee_model.calculate_taker_fee(price_cents=50, count=10)
        assert fee == 10
    
    def test_taker_fee_at_extreme_prices(self):
        """Taker fee at extreme prices (low potential payout)."""
        fee_model = KalshiOct2025FeeModel()
        
        # At 5 cents, potential payout = min(5, 95) = 5
        # Fee = ceil(5 * 1 * 0.02) = ceil(0.1) = 1 (rounds up)
        fee = fee_model.calculate_taker_fee(price_cents=5, count=1)
        assert fee == 1
        
        # At 95 cents, same calculation
        fee = fee_model.calculate_taker_fee(price_cents=95, count=1)
        assert fee == 1
    
    def test_fee_rounds_up(self):
        """Fee should always round up to nearest cent."""
        fee_model = KalshiOct2025FeeModel()
        
        # At 30 cents, potential payout = 30
        # Fee = ceil(30 * 1 * 0.02) = ceil(0.6) = 1
        fee = fee_model.calculate_taker_fee(price_cents=30, count=1)
        assert fee == 1


class TestExecutionEngine:
    """Tests for order execution."""
    
    @pytest.fixture
    def engine(self):
        """Create execution engine with fee model."""
        return ExecutionEngine(
            fee_model=KalshiOct2025FeeModel(),
            execution_mode="taker_only",
        )
    
    @pytest.fixture
    def sample_book(self):
        """Create a sample orderbook."""
        return OrderbookSnapshot(
            ts=datetime.now(),
            sequence_id=1,
            ticker="TEST",
            yes_bids=[
                OrderbookLevel(price_cents=45, size=100),
                OrderbookLevel(price_cents=40, size=200),
            ],
            no_bids=[
                OrderbookLevel(price_cents=52, size=50),
                OrderbookLevel(price_cents=50, size=150),
            ],
        )
    
    def test_market_buy_yes(self, engine, sample_book):
        """Test market buy of YES contracts."""
        order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.BUY,
            order_type=OrderType.MARKET,
            count=10,
        )
        
        result = engine.execute_order(order, sample_book, datetime.now())
        
        assert not result.rejected
        assert result.filled_count == 10
        # YES asks come from NO bids: 100-52=48 (50 size), 100-50=50 (150 size)
        # Best ask is 48 cents
        assert result.fills[0].price_cents == 48
    
    def test_market_sell_yes(self, engine, sample_book):
        """Test market sell of YES contracts."""
        order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.SELL,
            order_type=OrderType.MARKET,
            count=50,
        )
        
        result = engine.execute_order(order, sample_book, datetime.now())
        
        assert not result.rejected
        assert result.filled_count == 50
        # Selling YES at best YES bid = 45 cents
        assert result.fills[0].price_cents == 45
    
    def test_partial_fill_exhausts_depth(self, engine, sample_book):
        """Order larger than book depth gets partial fill."""
        order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.SELL,
            order_type=OrderType.MARKET,
            count=500,  # More than 100 + 200 available
        )
        
        result = engine.execute_order(order, sample_book, datetime.now())
        
        assert not result.rejected
        # Should fill 100 at 45, then 200 at 40 = 300 total
        assert result.filled_count == 300
        assert len(result.fills) == 2
    
    def test_limit_order_must_cross_in_taker_mode(self, engine, sample_book):
        """Limit orders that don't cross are rejected in taker-only mode."""
        # Try to buy YES at 40 (below best ask of 48)
        order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.BUY,
            order_type=OrderType.LIMIT,
            count=10,
            limit_price_cents=40,
        )
        
        result = engine.execute_order(order, sample_book, datetime.now())
        
        assert result.rejected
        assert "does not cross" in result.reject_reason.lower()
    
    def test_limit_order_crosses(self, engine, sample_book):
        """Limit order that crosses executes."""
        # Buy YES at 50 (above best ask of 48)
        order = Order(
            ticker="TEST",
            side=Side.YES,
            action=Action.BUY,
            order_type=OrderType.LIMIT,
            count=10,
            limit_price_cents=50,
        )
        
        result = engine.execute_order(order, sample_book, datetime.now())
        
        assert not result.rejected
        assert result.filled_count == 10
        # Fills at best ask (48), not at limit price
        assert result.fills[0].price_cents == 48


class TestOrder:
    """Tests for order validation."""
    
    def test_limit_order_requires_price(self):
        """Limit orders must specify limit price."""
        with pytest.raises(ValueError, match="limit_price_cents"):
            Order(
                ticker="TEST",
                side=Side.YES,
                action=Action.BUY,
                order_type=OrderType.LIMIT,
                count=10,
            )
    
    def test_count_must_be_positive(self):
        """Order count must be positive."""
        with pytest.raises(ValueError, match="positive"):
            Order(
                ticker="TEST",
                side=Side.YES,
                action=Action.BUY,
                order_type=OrderType.MARKET,
                count=0,
            )
