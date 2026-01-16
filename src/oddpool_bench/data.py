"""
Data loading utilities for episode data.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .types import (
    EpisodeMetadata,
    OrderbookSnapshot,
    OrderbookLevel,
    SettlementResult,
    Side,
)


@dataclass
class TradePrint:
    """A single trade from the trade tape."""
    ts: datetime
    trade_id: str
    ticker: str
    side: Side  # The contract side traded (YES or NO)
    taker_side: Side  # Direction of taker (YES = taker bought YES, NO = taker bought NO)
    price_cents: int
    count: int
    
    @property
    def passive_side(self) -> Side:
        """Side of the passive (maker) order that was consumed."""
        # Taker bought YES → consumed NO bids at (100 - price)
        # Taker bought NO → consumed YES bids at (100 - price)
        return Side.NO if self.taker_side == Side.YES else Side.YES
    
    @property
    def passive_price_cents(self) -> int:
        """Price level of the passive liquidity consumed."""
        return 100 - self.price_cents


def load_episode_metadata(episode_dir: Path) -> EpisodeMetadata:
    """Load episode metadata from metadata.json."""
    with open(episode_dir / "metadata.json") as f:
        data = json.load(f)
    return EpisodeMetadata.from_dict(data)


def load_settlements(episode_dir: Path) -> dict[str, SettlementResult]:
    """Load settlement results from settlement.json."""
    with open(episode_dir / "settlement.json") as f:
        data = json.load(f)
    
    settlements = {}
    # New format: dict keyed by ticker
    for ticker, item in data.items():
        result = item.get("result")
        settled_ts = item.get("settled_ts")
        
        # Skip unsettled markets
        if result is None or settled_ts is None:
            continue
            
        settlements[ticker] = SettlementResult(
            ticker=ticker,
            result=result,
            settled_ts=datetime.fromisoformat(str(settled_ts).replace("Z", "+00:00")),
        )
    return settlements


def parse_orderbook_row(row: dict) -> OrderbookSnapshot:
    """
    Parse a single orderbook row into an OrderbookSnapshot.
    
    Handles both parquet dict format and CSV format.
    """
    # Parse timestamp - handle both 'ts' and 'timestamp' column names
    ts = row.get("ts") or row.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    
    # Parse bids (could be JSON string or already parsed)
    yes_bids_raw = row.get("yes_bids") or []
    no_bids_raw = row.get("no_bids") or []
    
    if isinstance(yes_bids_raw, str):
        yes_bids_raw = json.loads(yes_bids_raw) if yes_bids_raw and yes_bids_raw != "[]" else []
    if isinstance(no_bids_raw, str):
        no_bids_raw = json.loads(no_bids_raw) if no_bids_raw and no_bids_raw != "[]" else []
    
    # Convert to OrderbookLevel objects
    def parse_levels(raw: list) -> list[OrderbookLevel]:
        levels = []
        for item in raw:
            # Handle both price_cents (new) and price (old) formats
            price = item.get("price_cents") or item.get("price", 0)
            size = item.get("size", 0)
            # Convert price to cents if it's a float < 1
            if isinstance(price, float) and price < 1:
                price_cents = int(round(price * 100))
            else:
                price_cents = int(price)
            if price_cents > 0 and size > 0:
                levels.append(OrderbookLevel(price_cents=price_cents, size=int(size)))
        # Sort by price descending (best bid first)
        levels.sort(key=lambda x: -x.price_cents)
        return levels
    
    return OrderbookSnapshot(
        ts=ts,
        sequence_id=int(row.get("sequence_id", 0)),
        ticker=row["ticker"],
        yes_bids=parse_levels(yes_bids_raw),
        no_bids=parse_levels(no_bids_raw),
    )


def load_orderbook_parquet(episode_dir: Path) -> Iterator[OrderbookSnapshot]:
    """Load orderbook data from parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required for parquet files: pip install pyarrow")
    
    table = pq.read_table(episode_dir / "orderbook.parquet")
    
    for row in table.to_pylist():
        yield parse_orderbook_row(row)


def load_orderbook_csv(episode_dir: Path) -> Iterator[OrderbookSnapshot]:
    """Load orderbook data from CSV file."""
    import csv
    
    csv_path = episode_dir / "orderbook.csv"
    if not csv_path.exists():
        csv_path = episode_dir / "orderbook.csv.gz"
        import gzip
        opener = gzip.open
    else:
        opener = open
    
    with opener(csv_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse JSON strings for bid lists
            if row.get("yes_bids"):
                row["yes_bids"] = json.loads(row["yes_bids"])
            else:
                row["yes_bids"] = []
            if row.get("no_bids"):
                row["no_bids"] = json.loads(row["no_bids"])
            else:
                row["no_bids"] = []
            
            yield parse_orderbook_row(row)


def load_orderbook_data(episode_dir: Path) -> list[OrderbookSnapshot]:
    """
    Load orderbook data from episode directory.
    
    Tries parquet first, then CSV.
    """
    parquet_path = episode_dir / "orderbook.parquet"
    if parquet_path.exists():
        return list(load_orderbook_parquet(episode_dir))
    
    csv_path = episode_dir / "orderbook.csv"
    csv_gz_path = episode_dir / "orderbook.csv.gz"
    if csv_path.exists() or csv_gz_path.exists():
        return list(load_orderbook_csv(episode_dir))
    
    raise FileNotFoundError(
        f"No orderbook data found in {episode_dir}. "
        "Expected orderbook.parquet or orderbook.csv"
    )


def parse_trade_row(row: dict) -> TradePrint:
    """Parse a single trade row into a TradePrint."""
    ts = row.get("ts") or row.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    
    # Parse side enum
    side_str = str(row.get("side", "yes")).lower()
    side = Side.YES if side_str == "yes" else Side.NO
    
    taker_side_str = str(row.get("taker_side", "yes")).lower()
    taker_side = Side.YES if taker_side_str == "yes" else Side.NO
    
    return TradePrint(
        ts=ts,
        trade_id=str(row.get("trade_id", "")),
        ticker=str(row["ticker"]),
        side=side,
        taker_side=taker_side,
        price_cents=int(row.get("price_cents", 0)),
        count=int(row.get("count", 0)),
    )


def load_trades_parquet(episode_dir: Path) -> Iterator[TradePrint]:
    """Load trade data from parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required for parquet files: pip install pyarrow")
    
    trades_path = episode_dir / "trades.parquet"
    if not trades_path.exists():
        return
    
    table = pq.read_table(trades_path)
    
    for row in table.to_pylist():
        yield parse_trade_row(row)


def load_trades_csv(episode_dir: Path) -> Iterator[TradePrint]:
    """Load trade data from CSV file."""
    import csv
    
    csv_path = episode_dir / "trades.csv"
    if not csv_path.exists():
        csv_path = episode_dir / "trades.csv.gz"
        if not csv_path.exists():
            return
        import gzip
        opener = gzip.open
    else:
        opener = open
    
    with opener(csv_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield parse_trade_row(row)


def load_trades_data(episode_dir: Path) -> list[TradePrint]:
    """
    Load trade tape data from episode directory.
    
    Returns empty list if no trades file exists.
    """
    parquet_path = episode_dir / "trades.parquet"
    if parquet_path.exists():
        return list(load_trades_parquet(episode_dir))
    
    csv_path = episode_dir / "trades.csv"
    csv_gz_path = episode_dir / "trades.csv.gz"
    if csv_path.exists() or csv_gz_path.exists():
        return list(load_trades_csv(episode_dir))
    
    # No trades file is valid - episode just doesn't have trade data
    return []


def has_trades_data(episode_dir: Path) -> bool:
    """Check if episode has trade tape data."""
    return (
        (episode_dir / "trades.parquet").exists() or
        (episode_dir / "trades.csv").exists() or
        (episode_dir / "trades.csv.gz").exists()
    )


def list_episodes(episodes_dir: Path) -> list[str]:
    """List available episode IDs in the episodes directory."""
    episodes = []
    for item in episodes_dir.iterdir():
        if item.is_dir() and (item / "metadata.json").exists():
            episodes.append(item.name)
    return sorted(episodes)
