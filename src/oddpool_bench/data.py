"""
Data loading utilities for episode data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .types import (
    EpisodeMetadata,
    OrderbookSnapshot,
    OrderbookLevel,
    SettlementResult,
)


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
    for item in data["settlements"]:
        settlements[item["ticker"]] = SettlementResult(
            ticker=item["ticker"],
            result=item["result"],
            settled_ts=datetime.fromisoformat(item["settled_ts"]),
        )
    return settlements


def parse_orderbook_row(row: dict) -> OrderbookSnapshot:
    """
    Parse a single orderbook row into an OrderbookSnapshot.
    
    Handles both parquet dict format and CSV format.
    """
    # Parse timestamp
    ts = row["timestamp"]
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    
    # Parse bids (could be JSON string or already parsed)
    yes_bids_raw = row.get("yes_bids") or []
    no_bids_raw = row.get("no_bids") or []
    
    if isinstance(yes_bids_raw, str):
        import ast
        yes_bids_raw = ast.literal_eval(yes_bids_raw) if yes_bids_raw else []
    if isinstance(no_bids_raw, str):
        import ast
        no_bids_raw = ast.literal_eval(no_bids_raw) if no_bids_raw else []
    
    # Convert to OrderbookLevel objects
    # Data has prices as floats (0.01-0.99), convert to cents
    def parse_levels(raw: list) -> list[OrderbookLevel]:
        levels = []
        for item in raw:
            price = item.get("price", 0)
            size = item.get("size", 0)
            # Convert price to cents if it's a float < 1
            if isinstance(price, float) and price < 1:
                price_cents = int(round(price * 100))
            else:
                price_cents = int(price)
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
    import json
    
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


def list_episodes(episodes_dir: Path) -> list[str]:
    """List available episode IDs in the episodes directory."""
    episodes = []
    for item in episodes_dir.iterdir():
        if item.is_dir() and (item / "metadata.json").exists():
            episodes.append(item.name)
    return sorted(episodes)
