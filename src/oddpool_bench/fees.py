"""
Fee models for the benchmark.

Implements Kalshi's fee schedule with versioning support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


class FeeModel(ABC):
    """Abstract base class for fee models."""
    
    @abstractmethod
    def calculate_taker_fee(self, price_cents: int, count: int) -> int:
        """
        Calculate taker fee in cents.
        
        Args:
            price_cents: Execution price per contract (1-99)
            count: Number of contracts
            
        Returns:
            Total fee in cents (rounded up)
        """
        pass
    
    @abstractmethod
    def calculate_maker_fee(self, price_cents: int, count: int) -> int:
        """
        Calculate maker fee in cents (for future use).
        
        Args:
            price_cents: Execution price per contract (1-99)
            count: Number of contracts
            
        Returns:
            Total fee in cents (rounded up)
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version string for this fee model."""
        pass


@dataclass
class KalshiOct2025FeeModel(FeeModel):
    """
    Kalshi fee model as of October 1, 2025.
    
    From Kalshi's published fee schedule:
    - Taker fee: 2% of potential payout, calculated as min(price, 100-price) * 0.02
    - Maker fee: 0.7% of potential payout (same calculation)
    - No settlement fee
    - Fees are rounded up to nearest cent
    
    Potential payout = min(price, 100-price) per contract
    This represents the maximum profit possible from the trade.
    """
    
    TAKER_FEE_RATE = 0.02  # 2%
    MAKER_FEE_RATE = 0.007  # 0.7%
    
    def _calculate_potential_payout(self, price_cents: int) -> int:
        """Calculate potential payout per contract."""
        return min(price_cents, 100 - price_cents)
    
    def calculate_taker_fee(self, price_cents: int, count: int) -> int:
        """
        Calculate taker fee.
        
        Fee = ceil(potential_payout * count * 0.02)
        """
        potential_payout = self._calculate_potential_payout(price_cents)
        fee_float = potential_payout * count * self.TAKER_FEE_RATE
        return math.ceil(fee_float)
    
    def calculate_maker_fee(self, price_cents: int, count: int) -> int:
        """
        Calculate maker fee.
        
        Fee = ceil(potential_payout * count * 0.007)
        """
        potential_payout = self._calculate_potential_payout(price_cents)
        fee_float = potential_payout * count * self.MAKER_FEE_RATE
        return math.ceil(fee_float)
    
    @property
    def version(self) -> str:
        return "kalshi_oct_2025"


# Registry of available fee models
FEE_MODELS: dict[str, type[FeeModel]] = {
    "kalshi_oct_2025": KalshiOct2025FeeModel,
}


def get_fee_model(version: str) -> FeeModel:
    """
    Get a fee model by version string.
    
    Args:
        version: Fee model version (e.g., "kalshi_oct_2025")
        
    Returns:
        Instantiated fee model
        
    Raises:
        ValueError: If version is not recognized
    """
    if version not in FEE_MODELS:
        raise ValueError(
            f"Unknown fee model version: {version}. "
            f"Available: {list(FEE_MODELS.keys())}"
        )
    return FEE_MODELS[version]()
