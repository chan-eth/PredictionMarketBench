"""
Core simulator for event-driven replay of prediction market episodes.

Supports both taker-only (v0) and maker+taker (v1) execution modes.
In maker mode, uses trade tape to determine maker fill timing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
import heapq

from .types import (
    Side,
    Order,
    OrderResult,
    Fill,
    FillType,
    RestingOrder,
    TimeInForce,
    OrderType,
    Position,
    OrderbookSnapshot,
    SettlementResult,
    EpisodeMetadata,
    EpisodeResult,
    EquitySnapshot,
    Action,
)
from .fees import FeeModel, get_fee_model
from .execution import ExecutionEngine
from .portfolio import Portfolio
from .agent import Agent, AgentContext
from .data import (
    load_episode_metadata,
    load_orderbook_data,
    load_settlements,
    load_trades_data,
    TradePrint,
)
from .metrics import compute_metrics
from .maker_queue import MakerQueueManager


@dataclass
class SimulatorConfig:
    """Configuration for the simulator."""
    
    # Agent calling policy
    agent_call_cadence_seconds: float = 5.0  # Call agent every N seconds
    
    # Equity curve sampling
    equity_sample_interval_seconds: float = 60.0  # Sample equity every N seconds
    
    # Tool budget per agent step
    max_tool_calls_per_step: int = 100
    
    # Maker queue mode: "trade_only" (conservative) or "reconciled"
    maker_queue_mode: str = "trade_only"
    
    # Whether to print progress
    verbose: bool = False


@dataclass
class TimelineEvent:
    """An event in the simulation timeline."""
    ts: datetime
    sequence_id: int
    event_type: str  # "orderbook_update", "trade_print", "settlement"
    ticker: Optional[str] = None
    data: Optional[any] = None
    
    def __lt__(self, other):
        """For heap ordering by timestamp then sequence."""
        if self.ts != other.ts:
            return self.ts < other.ts
        return self.sequence_id < other.sequence_id


class Simulator:
    """
    Event-driven replay simulator for prediction market episodes.
    
    Supports two execution modes:
    - taker_only: Orders must cross immediately or are rejected
    - maker_taker: GTC orders can rest and fill via trade tape
    """
    
    def __init__(
        self,
        episode_dir: Path,
        config: Optional[SimulatorConfig] = None,
    ):
        """
        Initialize simulator for an episode.
        
        Args:
            episode_dir: Path to episode directory
            config: Simulator configuration
        """
        self.episode_dir = Path(episode_dir)
        self.config = config or SimulatorConfig()
        
        # Load episode data
        self.metadata = load_episode_metadata(self.episode_dir)
        self.settlements = load_settlements(self.episode_dir)
        self.orderbook_data = load_orderbook_data(self.episode_dir)
        
        # Load trade tape if available (for maker fills)
        self.trade_data: list[TradePrint] = []
        if self.metadata.has_trades_tape:
            self.trade_data = load_trades_data(self.episode_dir)
        
        # Initialize fee model and execution engine
        self.fee_model = get_fee_model(self.metadata.fee_model_version)
        self.execution_engine = ExecutionEngine(
            fee_model=self.fee_model,
            execution_mode=self.metadata.execution_mode,
        )
        
        # Initialize maker queue manager (for maker_taker mode)
        self.maker_queue: Optional[MakerQueueManager] = None
        if self.metadata.execution_mode == "maker_taker":
            self.maker_queue = MakerQueueManager(self.fee_model)
        
        # Initialize state (reset in run())
        self.portfolio: Optional[Portfolio] = None
        self.current_ts: Optional[datetime] = None
        self.current_orderbooks: dict[str, OrderbookSnapshot] = {}
        self.equity_curve: list[EquitySnapshot] = []
        self.order_results: list[OrderResult] = []
        self.maker_fills: list[tuple[RestingOrder, Fill]] = []
    
    def _build_timeline(self) -> list[TimelineEvent]:
        """Build the global event timeline merging orderbooks and trades."""
        events = []
        seq = 0
        
        # Add orderbook updates
        for snapshot in self.orderbook_data:
            events.append(TimelineEvent(
                ts=snapshot.ts,
                sequence_id=seq,
                event_type="orderbook_update",
                ticker=snapshot.ticker,
                data=snapshot,
            ))
            seq += 1
        
        # Add trade prints (for maker fills)
        for trade in self.trade_data:
            events.append(TimelineEvent(
                ts=trade.ts,
                sequence_id=seq,
                event_type="trade_print",
                ticker=trade.ticker,
                data=trade,
            ))
            seq += 1
        
        # Add settlement events
        for ticker, settlement in self.settlements.items():
            events.append(TimelineEvent(
                ts=settlement.settled_ts,
                sequence_id=999999999,  # Settlement after all other events
                event_type="settlement",
                ticker=ticker,
                data=settlement,
            ))
        
        # Sort by timestamp, then sequence
        # Key ordering ensures: at same ts, orderbook updates come before trades
        # This is the anti-leakage rule: agent sees book state before trades at same ts
        events.sort(key=lambda e: (e.ts, 0 if e.event_type == "orderbook_update" else 1, e.sequence_id))
        
        return events
    
    def _should_call_agent(
        self,
        current_ts: datetime,
        last_agent_call: Optional[datetime],
    ) -> bool:
        """Check if we should call the agent based on cadence."""
        if last_agent_call is None:
            return True
        
        elapsed = (current_ts - last_agent_call).total_seconds()
        return elapsed >= self.config.agent_call_cadence_seconds
    
    def _should_sample_equity(
        self,
        current_ts: datetime,
        last_sample: Optional[datetime],
    ) -> bool:
        """Check if we should sample equity."""
        if last_sample is None:
            return True
        
        elapsed = (current_ts - last_sample).total_seconds()
        return elapsed >= self.config.equity_sample_interval_seconds
    
    def _create_agent_context(
        self,
        place_order_callback: Callable[[Order], OrderResult],
        cancel_order_callback: Callable[[str], bool],
    ) -> AgentContext:
        """Create context for agent with tool access."""
        # Get resting orders if in maker mode
        resting_orders = []
        if self.maker_queue:
            resting_orders = self.maker_queue.get_resting_orders()
        
        return AgentContext(
            current_ts=self.current_ts,
            orderbooks=self.current_orderbooks,
            positions=self.portfolio.positions,
            cash_cents=self.portfolio.cash_cents,
            equity_cents=self.portfolio.get_equity(self.current_orderbooks),
            tickers=self.metadata.tickers,
            event_slug=self.metadata.event_slug,
            end_ts=self.metadata.end_ts,
            place_order_callback=place_order_callback,
            cancel_order_callback=cancel_order_callback,
            observation_depth=self.metadata.observation_depth,
            resting_orders=resting_orders,
            execution_mode=self.metadata.execution_mode,
        )
    
    def _place_order(self, order: Order) -> OrderResult:
        """Process an order from the agent."""
        if order.ticker not in self.current_orderbooks:
            return OrderResult(
                order=order,
                ts=self.current_ts,
                filled_count=0,
                fills=[],
                total_cost_cents=0,
                total_fees_cents=0,
                rejected=True,
                reject_reason=f"No orderbook data for ticker: {order.ticker}",
            )
        
        orderbook = self.current_orderbooks[order.ticker]
        result = self.execution_engine.execute_order(order, orderbook, self.current_ts)
        
        # Apply taker fills to portfolio
        if result.fills:
            self.portfolio.apply_order_result(result, orderbook)
        
        # Handle resting orders (GTC or POST_ONLY that didn't fully fill)
        if (
            not result.rejected and
            self.maker_queue and
            order.time_in_force in (TimeInForce.GTC, TimeInForce.POST_ONLY) and
            order.order_type == OrderType.LIMIT
        ):
            remaining = order.count - result.filled_count
            if remaining > 0:
                # Check if order would cross (shouldn't rest if it would)
                would_cross = self._would_order_cross(order, orderbook)
                if not would_cross:
                    # Create resting order
                    resting = self.maker_queue.place_order(order, orderbook, self.current_ts)
                    # Update count to remaining
                    resting.remaining_count = remaining
                    result.resting_order = resting
        
        # Track result
        self.order_results.append(result)
        
        return result
    
    def _would_order_cross(self, order: Order, orderbook: OrderbookSnapshot) -> bool:
        """Check if a limit order would cross the book."""
        is_buy = order.action == Action.BUY
        
        if order.side == Side.YES:
            if is_buy:
                best_ask = orderbook.yes_best_ask
                if best_ask is not None and order.limit_price_cents >= best_ask:
                    return True
            else:
                best_bid = orderbook.yes_best_bid
                if best_bid is not None and order.limit_price_cents <= best_bid:
                    return True
        else:
            if is_buy:
                best_ask = orderbook.no_best_ask
                if best_ask is not None and order.limit_price_cents >= best_ask:
                    return True
            else:
                best_bid = orderbook.no_best_bid
                if best_bid is not None and order.limit_price_cents <= best_bid:
                    return True
        
        return False
    
    def _cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order."""
        if not self.maker_queue:
            return False
        
        canceled = self.maker_queue.cancel_order(order_id)
        return canceled is not None
    
    def _process_trade_print(self, trade: TradePrint) -> None:
        """Process a trade print for maker fills."""
        if not self.maker_queue:
            return
        
        # Get fills from queue
        fills = self.maker_queue.process_trade(
            ticker=trade.ticker,
            taker_side=trade.taker_side,
            trade_price_cents=trade.price_cents,
            volume=trade.count,
            ts=trade.ts,
        )
        
        # Apply maker fills to portfolio
        for resting, fill in fills:
            self._apply_maker_fill(resting, fill)
            self.maker_fills.append((resting, fill))
    
    def _apply_maker_fill(self, resting: RestingOrder, fill: Fill) -> None:
        """Apply a maker fill to the portfolio."""
        original = resting.original_order
        
        # Create a synthetic OrderResult to use portfolio's apply_order_result
        synthetic_result = OrderResult(
            order=original,
            ts=fill.fill_ts or self.current_ts,
            filled_count=fill.size,
            fills=[fill],
            total_cost_cents=fill.price_cents * fill.size + fill.fee_cents if original.action == Action.BUY else -(fill.price_cents * fill.size - fill.fee_cents),
            total_fees_cents=fill.fee_cents,
            rejected=False,
        )
        
        # Apply to portfolio (this handles position updates and cash)
        self.portfolio.apply_order_result(synthetic_result, orderbook=None)
    
    def run(self, agent: Agent) -> EpisodeResult:
        """
        Run the simulation with the given agent.
        
        Args:
            agent: Agent instance to run
            
        Returns:
            EpisodeResult with performance metrics
        """
        # Initialize state
        self.portfolio = Portfolio(cash_cents=self.metadata.initial_bankroll_cents)
        self.current_ts = self.metadata.start_ts
        self.current_orderbooks = {}
        self.equity_curve = []
        self.order_results = []
        self.maker_fills = []
        
        # Reset maker queue
        if self.maker_queue:
            self.maker_queue = MakerQueueManager(self.fee_model)
        
        # Notify agent of episode start
        agent.on_episode_start(self.metadata.to_dict())
        
        # Build timeline
        timeline = self._build_timeline()
        
        if self.config.verbose:
            print(f"Running episode {self.metadata.episode_id}")
            print(f"  {len(self.orderbook_data)} orderbook snapshots")
            print(f"  {len(self.trade_data)} trade prints")
            print(f"  {len(self.metadata.tickers)} tickers")
            print(f"  Mode: {self.metadata.execution_mode}")
        
        # Tracking
        last_agent_call: Optional[datetime] = None
        last_equity_sample: Optional[datetime] = None
        settled_tickers: set[str] = set()
        
        # Initial equity sample
        initial_equity = self.metadata.initial_bankroll_cents
        self.equity_curve.append(EquitySnapshot(
            ts=self.metadata.start_ts,
            cash_cents=initial_equity,
            position_value_cents=0,
            equity_cents=initial_equity,
        ))
        last_equity_sample = self.metadata.start_ts
        
        # Process events
        for event in timeline:
            self.current_ts = event.ts
            
            if event.event_type == "orderbook_update":
                # Update orderbook state
                snapshot: OrderbookSnapshot = event.data
                self.current_orderbooks[snapshot.ticker] = snapshot
                
                # Optionally update maker queue with new book state
                if self.maker_queue:
                    self.maker_queue.update_env_from_snapshot(
                        snapshot, 
                        mode=self.config.maker_queue_mode
                    )
                
            elif event.event_type == "trade_print":
                # Process trade for maker fills
                trade: TradePrint = event.data
                self._process_trade_print(trade)
                
            elif event.event_type == "settlement":
                # Process settlement
                settlement: SettlementResult = event.data
                
                # Cancel any resting orders for this ticker
                if self.maker_queue:
                    for resting in self.maker_queue.get_resting_orders(settlement.ticker):
                        self.maker_queue.cancel_order(resting.order_id)
                
                self.portfolio.settle_position(settlement.ticker, settlement)
                settled_tickers.add(settlement.ticker)
                
                if self.config.verbose:
                    print(f"  Settled {settlement.ticker}: {settlement.result}")
            
            # Check if we should call agent (after orderbook updates, before trades at same ts)
            if event.event_type == "orderbook_update":
                if self._should_call_agent(self.current_ts, last_agent_call):
                    ctx = self._create_agent_context(self._place_order, self._cancel_order)
                    try:
                        agent.act(ctx)
                    except Exception as e:
                        if self.config.verbose:
                            print(f"  Agent error: {e}")
                    last_agent_call = self.current_ts
            
            # Sample equity
            if self._should_sample_equity(self.current_ts, last_equity_sample):
                self.equity_curve.append(
                    self.portfolio.get_equity_snapshot(self.current_ts, self.current_orderbooks)
                )
                last_equity_sample = self.current_ts
        
        # Ensure all tickers are settled
        for ticker in self.metadata.tickers:
            if ticker not in settled_tickers and ticker in self.settlements:
                settlement = self.settlements[ticker]
                self.portfolio.settle_position(ticker, settlement)
        
        # Final equity sample
        final_equity = self.portfolio.get_equity(self.current_orderbooks)
        self.equity_curve.append(EquitySnapshot(
            ts=self.current_ts,
            cash_cents=self.portfolio.cash_cents,
            position_value_cents=0,  # All positions settled
            equity_cents=self.portfolio.cash_cents,
        ))
        
        # Compute metrics
        metrics = compute_metrics(
            initial_equity=initial_equity,
            final_equity=final_equity,
            equity_curve=self.equity_curve,
            total_contracts=self.portfolio.total_contracts_traded,
            total_notional=self.portfolio.total_notional_cents,
            total_fees=self.portfolio.total_fees_paid_cents,
            total_slippage=self.portfolio.total_slippage_cents,
            fill_ratio=self.portfolio.fill_ratio,
        )
        
        result = EpisodeResult(
            episode_id=self.metadata.episode_id,
            initial_equity_cents=initial_equity,
            final_equity_cents=final_equity,
            total_pnl_cents=metrics.total_pnl_cents,
            total_pnl_pct=metrics.total_pnl_pct,
            max_drawdown_pct=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            total_contracts_traded=metrics.total_contracts_traded,
            total_notional_cents=metrics.total_notional_cents,
            total_fees_cents=metrics.total_fees_cents,
            total_slippage_cents=metrics.total_slippage_cents,
            fill_ratio=metrics.fill_ratio,
            equity_curve=self.equity_curve,
            order_results=self.order_results,
            settlements=self.settlements,
        )
        
        # Notify agent
        agent.on_episode_end({
            "episode_id": result.episode_id,
            "total_pnl_cents": result.total_pnl_cents,
            "total_pnl_pct": result.total_pnl_pct,
        })
        
        if self.config.verbose:
            print(f"  PnL: ${result.total_pnl_cents / 100:.2f} ({result.total_pnl_pct * 100:.2f}%)")
            print(f"  Contracts traded: {result.total_contracts_traded}")
            print(f"  Fees: ${result.total_fees_cents / 100:.2f}")
            if self.maker_fills:
                print(f"  Maker fills: {len(self.maker_fills)}")
        
        return result
