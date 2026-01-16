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
    
    Trading fees are charged as a variable percentage fee of the expected earnings 
    on an individual contract, which is calculated by multiplying the maximum 
    potential earnings from the contract by the implied probability of making 
    those earnings, or the price of the contract divided by $1.
    
    Taker Fees:
        fee = round_up(0.07 × C × P × (1-P))
        
    Maker Fees:
        fee = round_up(0.0175 × C × P × (1-P))
        
    Where:
        P = the price of a contract in dollars (50 cents = 0.50)
        C = the number of contracts being traded
        round_up = rounds to the next cent
    
    The P × (1-P) formula means fees are highest at P=0.50 (50%) and decrease
    toward the extremes (P=0 or P=1).
    """
    
    TAKER_FEE_RATE = 0.07   # 7%
    MAKER_FEE_RATE = 0.0175  # 1.75%
    
    def calculate_taker_fee(self, price_cents: int, count: int) -> int:
        """
        Calculate taker fee.
        
        Fee = ceil(0.07 × C × P × (1-P))
        Where P is price in dollars (e.g., 50 cents = 0.50)
        
        Example: Buy 10 contracts at 50c
            P = 0.50
            fee = ceil(0.07 × 10 × 0.50 × 0.50) = ceil(0.175) = 1 cent
        """
        p = price_cents / 100.0  # Convert cents to dollars
        fee_float = self.TAKER_FEE_RATE * count * p * (1 - p)
        # Fee is already in dollars, convert to cents
        fee_cents = fee_float * 100
        return math.ceil(fee_cents)
    
    def calculate_maker_fee(self, price_cents: int, count: int) -> int:
        """
        Calculate maker fee.
        
        Fee = ceil(0.0175 × C × P × (1-P))
        Where P is price in dollars (e.g., 50 cents = 0.50)
        
        This is 1/4 of the taker fee, encouraging liquidity provision.
        
        Example: Provide 10 contracts at 50c
            P = 0.50  
            fee = ceil(0.0175 × 10 × 0.50 × 0.50) = ceil(0.04375) = 1 cent
        """
        p = price_cents / 100.0  # Convert cents to dollars
        fee_float = self.MAKER_FEE_RATE * count * p * (1 - p)
        # Fee is already in dollars, convert to cents
        fee_cents = fee_float * 100
        return math.ceil(fee_cents)
    
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
