"""
Example agents for the Oddpool PredictionMarketBench.

These demonstrate how to implement agents for the benchmark.
"""

from oddpool_bench import (
    Agent,
    AgentContext,
    Order,
    Side,
    Action,
    OrderType,
    TimeInForce,
)


class PassiveAgent(Agent):
    """
    A passive agent that never trades.
    
    Useful as a baseline to compare against.
    """
    
    def act(self, ctx: AgentContext) -> None:
        """Do nothing."""
        pass


class RandomAgent(Agent):
    """
    An agent that makes random trades.
    
    This is a simple example to demonstrate the agent interface.
    Not expected to be profitable!
    """
    
    def __init__(self, trade_probability: float = 0.1, max_position: int = 10):
        """
        Args:
            trade_probability: Probability of trading on any given step
            max_position: Maximum position size per ticker
        """
        self.trade_probability = trade_probability
        self.max_position = max_position
        self._rng = None
    
    def on_episode_start(self, metadata: dict) -> None:
        """Initialize RNG with episode-specific seed for reproducibility."""
        import random
        # Use episode ID as seed for reproducibility
        seed = hash(metadata.get("episode_id", "")) % (2**32)
        self._rng = random.Random(seed)
    
    def act(self, ctx: AgentContext) -> None:
        """Make random trading decisions."""
        if self._rng is None:
            import random
            self._rng = random.Random()
        
        if self._rng.random() > self.trade_probability:
            return
        
        # Get markets and positions
        markets = ctx.get_markets()
        positions = ctx.get_positions()
        cash = ctx.get_cash()
        
        if cash["cash_cents"] < 100:  # Need at least $1
            return
        
        # Pick a random market with quotes
        active_markets = [
            m for m in markets
            if m.yes_best_bid is not None and m.yes_best_ask is not None
        ]
        
        if not active_markets:
            return
        
        market = self._rng.choice(active_markets)
        ticker = market.ticker
        
        # Get current position
        pos = positions.get(ticker, {})
        yes_pos = pos.get("yes_contracts", 0)
        no_pos = pos.get("no_contracts", 0)
        
        # Decide action based on position and randomness
        if self._rng.random() < 0.5:
            # Trade YES side
            side = Side.YES
            if yes_pos < self.max_position:
                action = Action.BUY
                price = market.yes_best_ask
            elif yes_pos > -self.max_position:
                action = Action.SELL
                price = market.yes_best_bid
            else:
                return
        else:
            # Trade NO side
            side = Side.NO
            if no_pos < self.max_position:
                action = Action.BUY
                price = market.no_best_ask
            elif no_pos > -self.max_position:
                action = Action.SELL
                price = market.no_best_bid
            else:
                return
        
        if price is None:
            return
        
        # Place a small market order
        order = Order(
            ticker=ticker,
            side=side,
            action=action,
            order_type=OrderType.MARKET,
            count=self._rng.randint(1, 3),
        )
        
        result = ctx.place_order(order)
        # Result is available if needed for logging


class SpreadCapture(Agent):
    """
    A simple spread-capturing agent.
    
    Looks for markets with wide spreads and tries to capture the spread
    by buying at the bid and selling at the ask (as a taker).
    
    Note: In taker-only mode, this can only capture spread by round-tripping
    immediately, which is generally not profitable after fees.
    """
    
    def __init__(self, min_spread_cents: int = 10, position_limit: int = 5):
        """
        Args:
            min_spread_cents: Minimum spread to consider trading
            position_limit: Maximum position per side per ticker
        """
        self.min_spread_cents = min_spread_cents
        self.position_limit = position_limit
    
    def act(self, ctx: AgentContext) -> None:
        """Look for spread opportunities."""
        markets = ctx.get_markets()
        positions = ctx.get_positions()
        
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            
            spread = market.yes_best_ask - market.yes_best_bid
            
            if spread >= self.min_spread_cents:
                # Wide spread - consider a trade
                pos = positions.get(market.ticker, {})
                yes_pos = pos.get("yes_contracts", 0)
                
                # If we have no position, buy some
                if yes_pos == 0:
                    order = Order(
                        ticker=market.ticker,
                        side=Side.YES,
                        action=Action.BUY,
                        order_type=OrderType.LIMIT,
                        count=1,
                        limit_price_cents=market.yes_best_ask,
                    )
                    ctx.place_order(order)
                
                # If we have a long position, try to sell at the ask
                elif yes_pos > 0:
                    # In taker-only mode, we can only sell at the bid
                    # This would lock in a loss, so skip
                    pass


class MomentumAgent(Agent):
    """
    A simple momentum-following agent.
    
    Tracks price changes and trades in the direction of momentum.
    """
    
    def __init__(self, lookback_steps: int = 10, threshold_cents: int = 3):
        """
        Args:
            lookback_steps: Number of steps to track for momentum
            threshold_cents: Minimum price change to trigger trade
        """
        self.lookback_steps = lookback_steps
        self.threshold_cents = threshold_cents
        self.price_history: dict[str, list[int]] = {}
    
    def on_episode_start(self, metadata: dict) -> None:
        """Reset price history."""
        self.price_history = {}
    
    def act(self, ctx: AgentContext) -> None:
        """Trade based on momentum signals."""
        markets = ctx.get_markets()
        positions = ctx.get_positions()
        cash = ctx.get_cash()
        
        for market in markets:
            if market.yes_best_bid is None or market.yes_best_ask is None:
                continue
            
            ticker = market.ticker
            mid = (market.yes_best_bid + market.yes_best_ask) // 2
            
            # Update history
            if ticker not in self.price_history:
                self.price_history[ticker] = []
            
            history = self.price_history[ticker]
            history.append(mid)
            
            # Keep only lookback window
            if len(history) > self.lookback_steps:
                history.pop(0)
            
            # Need enough history
            if len(history) < self.lookback_steps:
                continue
            
            # Calculate momentum
            price_change = history[-1] - history[0]
            
            pos = positions.get(ticker, {})
            yes_pos = pos.get("yes_contracts", 0)
            
            # Trade on momentum
            if price_change >= self.threshold_cents and yes_pos < 5:
                # Upward momentum - buy YES
                if cash["cash_cents"] >= market.yes_best_ask * 2:
                    order = Order(
                        ticker=ticker,
                        side=Side.YES,
                        action=Action.BUY,
                        order_type=OrderType.MARKET,
                        count=1,
                    )
                    ctx.place_order(order)
            
            elif price_change <= -self.threshold_cents and yes_pos > 0:
                # Downward momentum - sell YES position
                order = Order(
                    ticker=ticker,
                    side=Side.YES,
                    action=Action.SELL,
                    order_type=OrderType.MARKET,
                    count=min(1, yes_pos),
                )
                ctx.place_order(order)
