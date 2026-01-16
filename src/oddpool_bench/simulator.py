"""
Core simulator for event-driven replay of prediction market episodes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
import heapq

from .types import (
    Order,
    OrderResult,
    Position,
    OrderbookSnapshot,
    SettlementResult,
    EpisodeMetadata,
    EpisodeResult,
    EquitySnapshot,
)
from .fees import FeeModel, get_fee_model
from .execution import ExecutionEngine
from .portfolio import Portfolio
from .agent import Agent, AgentContext
from .data import load_episode_metadata, load_orderbook_data, load_settlements
from .metrics import compute_metrics


@dataclass
class SimulatorConfig:
    """Configuration for the simulator."""
    
    # Agent calling policy
    agent_call_cadence_seconds: float = 5.0  # Call agent every N seconds
    
    # Equity curve sampling
    equity_sample_interval_seconds: float = 60.0  # Sample equity every N seconds
    
    # Tool budget per agent step
    max_tool_calls_per_step: int = 100
    
    # Whether to print progress
    verbose: bool = False


@dataclass
class TimelineEvent:
    """An event in the simulation timeline."""
    ts: datetime
    sequence_id: int
    event_type: str  # "orderbook_update", "agent_call", "settlement"
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
    
    Replays orderbook data and invokes agent at configured cadence.
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
        
        # Initialize fee model and execution engine
        self.fee_model = get_fee_model(self.metadata.fee_model_version)
        self.execution_engine = ExecutionEngine(
            fee_model=self.fee_model,
            execution_mode=self.metadata.execution_mode,
        )
        
        # Initialize state (reset in run())
        self.portfolio: Optional[Portfolio] = None
        self.current_ts: Optional[datetime] = None
        self.current_orderbooks: dict[str, OrderbookSnapshot] = {}
        self.equity_curve: list[EquitySnapshot] = []
        self.order_results: list[OrderResult] = []
    
    def _build_timeline(self) -> list[TimelineEvent]:
        """Build the global event timeline."""
        events = []
        
        # Add orderbook updates
        for snapshot in self.orderbook_data:
            events.append(TimelineEvent(
                ts=snapshot.ts,
                sequence_id=snapshot.sequence_id,
                event_type="orderbook_update",
                ticker=snapshot.ticker,
                data=snapshot,
            ))
        
        # Add settlement events
        for ticker, settlement in self.settlements.items():
            events.append(TimelineEvent(
                ts=settlement.settled_ts,
                sequence_id=999999999,  # Settlement after all orderbook updates
                event_type="settlement",
                ticker=ticker,
                data=settlement,
            ))
        
        # Sort by timestamp, then sequence
        events.sort(key=lambda e: (e.ts, e.sequence_id))
        
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
    ) -> AgentContext:
        """Create context for agent with tool access."""
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
            observation_depth=self.metadata.observation_depth,
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
        
        # Apply to portfolio
        self.portfolio.apply_order_result(result, orderbook)
        
        # Track result
        self.order_results.append(result)
        
        return result
    
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
        
        # Notify agent of episode start
        agent.on_episode_start(self.metadata.to_dict())
        
        # Build timeline
        timeline = self._build_timeline()
        
        if self.config.verbose:
            print(f"Running episode {self.metadata.episode_id}")
            print(f"  {len(timeline)} events, {len(self.metadata.tickers)} tickers")
        
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
                
            elif event.event_type == "settlement":
                # Process settlement
                settlement: SettlementResult = event.data
                self.portfolio.settle_position(settlement.ticker, settlement)
                settled_tickers.add(settlement.ticker)
                
                if self.config.verbose:
                    print(f"  Settled {settlement.ticker}: {settlement.result}")
            
            # Check if we should call agent (not during/after settlements)
            if event.event_type == "orderbook_update":
                if self._should_call_agent(self.current_ts, last_agent_call):
                    ctx = self._create_agent_context(self._place_order)
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
        
        return result
