#!/usr/bin/env python3
"""
Episode Recorder — captures live Kalshi market data into backtest episode format.

Usage:
    python record_episode.py --event KXHIGHNY-26MAR15
    python record_episode.py --series KXHIGHNY
    python record_episode.py --series KXHIGHNY --series KXBTCD
    python record_episode.py --series KXHIGHNY --output ./episodes/
"""

import argparse
import asyncio
import base64
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

# ── Kalshi API constants ──────────────────────────────────────────────────────

BASE_URL = "https://api.elections.kalshi.com"
BASE_PATH = "/trade-api/v2"

# ── Auth ──────────────────────────────────────────────────────────────────────


def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def sign_request(private_key, api_key: str, method: str, path: str) -> dict:
    ts_ms = str(int(time.time() * 1000))
    message = f"{ts_ms}{method}{path}".encode()
    sig = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
    }


# ── API helpers ───────────────────────────────────────────────────────────────


class KalshiAPI:
    def __init__(self, api_key: str, private_key):
        self.api_key = api_key
        self.private_key = private_key
        self.client = httpx.AsyncClient(base_url=BASE_URL, timeout=15.0)

    async def close(self):
        await self.client.aclose()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        full_path = BASE_PATH + path
        headers = sign_request(self.private_key, self.api_key, "GET", full_path)
        headers["Accept"] = "application/json"
        resp = await self.client.get(full_path, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        ticker: str | None = None,
    ) -> list[dict]:
        """Fetch markets with cursor-based pagination."""
        all_markets = []
        cursor = None
        while True:
            params = {"limit": "100"}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if event_ticker:
                params["event_ticker"] = event_ticker
            if status:
                params["status"] = status
            if ticker:
                params["ticker"] = ticker
            if cursor:
                params["cursor"] = cursor
            data = await self._get("/markets", params)
            all_markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return all_markets

    async def get_orderbook(self, ticker: str) -> dict:
        data = await self._get(f"/markets/{ticker}/orderbook")
        return data.get("orderbook_fp", data.get("orderbook", {}))

    async def get_trades(
        self, ticker: str, cursor: str | None = None, limit: int = 100
    ) -> tuple[list[dict], str | None]:
        params = {"limit": str(limit)}
        if cursor:
            params["cursor"] = cursor
        data = await self._get(f"/markets/{ticker}/trades", params)
        return data.get("trades", []), data.get("cursor")


# ── Orderbook conversion ─────────────────────────────────────────────────────


def convert_levels(dollar_levels: list[list[str]]) -> str:
    """Convert Kalshi API orderbook levels to our JSON format.

    API format:  [["0.45", "100.00"], ...]
    Our format:  [{"price_cents": 45, "size": 100}, ...]
    """
    result = []
    for level in dollar_levels:
        if len(level) >= 2:
            price_cents = int(round(float(level[0]) * 100))
            size = int(round(float(level[1])))
            if price_cents > 0 and size > 0:
                result.append({"price_cents": price_cents, "size": size})
    # Sort best bid first (descending)
    result.sort(key=lambda x: -x["price_cents"])
    return json.dumps(result)


# ── Parquet schemas ───────────────────────────────────────────────────────────

ORDERBOOK_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("sequence_id", pa.int64()),
        ("ticker", pa.string()),
        ("yes_bids", pa.string()),
        ("no_bids", pa.string()),
    ]
)

TRADES_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("trade_id", pa.string()),
        ("ticker", pa.string()),
        ("side", pa.string()),
        ("taker_side", pa.string()),
        ("price_cents", pa.int64()),
        ("count", pa.int64()),
    ]
)


# ── Episode Recorder ─────────────────────────────────────────────────────────


class EpisodeRecorder:
    def __init__(self, api: KalshiAPI, event_ticker: str, output_dir: Path):
        self.api = api
        self.event_ticker = event_ticker
        self.output_dir = output_dir
        self.tickers: list[str] = []
        self.markets: dict[str, dict] = {}  # ticker -> market info
        self.orderbook_rows: list[dict] = []
        self.trade_rows: list[dict] = []
        self.last_trade_id: dict[str, str] = {}  # ticker -> last seen trade_id
        self.sequence_id = 0
        self.start_ts: datetime | None = None
        self.shutting_down = False
        self.flush_threshold = 1000

    async def discover_markets(self):
        """Find all tickers in this event."""
        markets = await self.api.get_markets(event_ticker=self.event_ticker)
        if not markets:
            raise ValueError(f"No markets found for event {self.event_ticker}")

        self.markets = {m["ticker"]: m for m in markets}
        self.tickers = sorted(self.markets.keys())
        print(f"  Found {len(self.tickers)} tickers:")
        for t in self.tickers:
            status = self.markets[t].get("status", "?")
            title = self.markets[t].get("subtitle", self.markets[t].get("title", ""))
            print(f"    {t}  [{status}]  {title}")

    async def poll_orderbooks(self):
        """Poll orderbooks for all tickers every 5s."""
        n = len(self.tickers)
        stagger = 5.0 / max(n, 1)  # spread requests across the window

        while not self.shutting_down:
            cycle_start = time.monotonic()
            for i, ticker in enumerate(self.tickers):
                if self.shutting_down:
                    return
                try:
                    ob = await self.api.get_orderbook(ticker)
                    now = datetime.now(timezone.utc)
                    if self.start_ts is None:
                        self.start_ts = now

                    self.sequence_id += 1
                    yes_bids = convert_levels(ob.get("yes_dollars", ob.get("yes", [])))
                    no_bids = convert_levels(ob.get("no_dollars", ob.get("no", [])))

                    self.orderbook_rows.append(
                        {
                            "ts": now,
                            "sequence_id": self.sequence_id,
                            "ticker": ticker,
                            "yes_bids": yes_bids,
                            "no_bids": no_bids,
                        }
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        pass  # market may have settled
                    else:
                        print(f"  [WARN] Orderbook {ticker}: {e.response.status_code}")
                except Exception as e:
                    print(f"  [WARN] Orderbook {ticker}: {e}")

                # Stagger to stay under rate limit
                if i < n - 1:
                    await asyncio.sleep(stagger)

            # Flush if buffer is large
            if len(self.orderbook_rows) >= self.flush_threshold:
                self._flush_orderbooks()

            # Wait remainder of 5s window
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, 5.0 - elapsed)
            if sleep_time > 0 and not self.shutting_down:
                await asyncio.sleep(sleep_time)

    async def poll_trades(self):
        """Poll trades for all tickers every 10s."""
        while not self.shutting_down:
            cycle_start = time.monotonic()
            for ticker in self.tickers:
                if self.shutting_down:
                    return
                try:
                    trades, _ = await self.api.get_trades(ticker)
                    last_seen = self.last_trade_id.get(ticker)
                    new_trades = []
                    for t in trades:
                        tid = t.get("trade_id", "")
                        if tid == last_seen:
                            break
                        new_trades.append(t)

                    if new_trades:
                        self.last_trade_id[ticker] = new_trades[0]["trade_id"]

                    for t in reversed(new_trades):  # append in chronological order
                        ts_str = t.get("created_time", "")
                        try:
                            ts = datetime.fromisoformat(
                                ts_str.replace("Z", "+00:00")
                            )
                        except (ValueError, AttributeError):
                            ts = datetime.now(timezone.utc)

                        yes_price = t.get("yes_price", 0)
                        taker_side = (t.get("taker_side") or "yes").lower()

                        self.trade_rows.append(
                            {
                                "ts": ts,
                                "trade_id": t.get("trade_id", ""),
                                "ticker": ticker,
                                "side": "yes",  # Kalshi trades are always quoted in YES
                                "taker_side": taker_side,
                                "price_cents": int(yes_price),
                                "count": int(t.get("count", 0)),
                            }
                        )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        print(f"  [WARN] Trades {ticker}: {e.response.status_code}")
                except Exception as e:
                    print(f"  [WARN] Trades {ticker}: {e}")

                await asyncio.sleep(0.15)  # small gap between tickers

            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, 10.0 - elapsed)
            if sleep_time > 0 and not self.shutting_down:
                await asyncio.sleep(sleep_time)

    async def wait_for_settlement(self):
        """After close_time, poll until all markets are settled."""
        # Find the latest close_time
        close_times = []
        for m in self.markets.values():
            ct = m.get("close_time") or m.get("expiration_time")
            if ct:
                try:
                    close_times.append(
                        datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    )
                except ValueError:
                    pass

        if not close_times:
            print("  [WARN] No close_time found, skipping settlement wait")
            return {}

        latest_close = max(close_times)
        now = datetime.now(timezone.utc)

        if now < latest_close:
            wait_secs = (latest_close - now).total_seconds()
            print(
                f"  Markets close at {latest_close.isoformat()}, "
                f"waiting {wait_secs/3600:.1f}h..."
            )
            # Continue recording until close
            while datetime.now(timezone.utc) < latest_close and not self.shutting_down:
                await asyncio.sleep(5)

        print("  Markets closed, polling for settlement...")
        settlements = {}
        unsettled = set(self.tickers)
        max_retries = 120  # 2 hours of polling at 60s

        for attempt in range(max_retries):
            if self.shutting_down or not unsettled:
                break

            try:
                for ticker in list(unsettled):
                    markets = await self.api.get_markets(ticker=ticker)
                    for m in markets:
                        if m.get("ticker") == ticker and m.get("result"):
                            result = m["result"].upper()
                            settled_ts = (
                                m.get("expiration_time")
                                or m.get("close_time")
                                or datetime.now(timezone.utc).isoformat()
                            )
                            settlements[ticker] = {
                                "result": result,
                                "settled_ts": settled_ts,
                            }
                            unsettled.discard(ticker)
                            print(f"    {ticker} → {result}")
            except Exception as e:
                print(f"  [WARN] Settlement poll: {e}")

            if unsettled:
                remaining = len(unsettled)
                print(
                    f"  Settlement poll {attempt+1}: "
                    f"{len(settlements)}/{len(self.tickers)} settled, "
                    f"{remaining} remaining"
                )
                await asyncio.sleep(60)

        return settlements

    def _flush_orderbooks(self):
        """Flush orderbook buffer to a temp parquet file (crash safety)."""
        if not self.orderbook_rows:
            return
        n = len(self.orderbook_rows)
        tmp_path = self.output_dir / f"_orderbook_partial_{self.sequence_id}.parquet"
        table = pa.table(
            {col: [r[col] for r in self.orderbook_rows] for col in ORDERBOOK_SCHEMA.names},
            schema=ORDERBOOK_SCHEMA,
        )
        pq.write_table(table, tmp_path)
        self.orderbook_rows.clear()
        print(f"  [FLUSH] {n} orderbook rows → {tmp_path.name}")

    def write_outputs(self, settlements: dict):
        """Write all output files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Merge any partial orderbook flushes + remaining buffer ──
        partial_files = sorted(self.output_dir.glob("_orderbook_partial_*.parquet"))
        tables = []
        for pf in partial_files:
            tables.append(pq.read_table(pf, schema=ORDERBOOK_SCHEMA))
        if self.orderbook_rows:
            tables.append(
                pa.table(
                    {col: [r[col] for r in self.orderbook_rows] for col in ORDERBOOK_SCHEMA.names},
                    schema=ORDERBOOK_SCHEMA,
                )
            )

        if tables:
            merged = pa.concat_tables(tables)
            pq.write_table(merged, self.output_dir / "orderbook.parquet")
            print(f"  Wrote orderbook.parquet ({merged.num_rows} rows)")
        else:
            print("  [WARN] No orderbook data collected")

        # Clean up partials
        for pf in partial_files:
            pf.unlink()

        # ── Trades ──
        if self.trade_rows:
            table = pa.table(
                {col: [r[col] for r in self.trade_rows] for col in TRADES_SCHEMA.names},
                schema=TRADES_SCHEMA,
            )
            pq.write_table(table, self.output_dir / "trades.parquet")
            print(f"  Wrote trades.parquet ({table.num_rows} rows)")
        else:
            # Write empty trades file so has_trades_tape is accurate
            table = pa.table(
                {col: [] for col in TRADES_SCHEMA.names},
                schema=TRADES_SCHEMA,
            )
            pq.write_table(table, self.output_dir / "trades.parquet")
            print("  Wrote trades.parquet (0 rows)")

        # ── Settlement ──
        with open(self.output_dir / "settlement.json", "w") as f:
            json.dump(settlements, f, indent=2)
        print(f"  Wrote settlement.json ({len(settlements)} tickers)")

        # ── Metadata ──
        end_ts = datetime.now(timezone.utc)
        meta = {
            "episode_id": self.event_ticker,
            "event_slug": self.event_ticker,
            "tickers": self.tickers,
            "start_ts": (
                self.start_ts.isoformat() if self.start_ts else end_ts.isoformat()
            ),
            "end_ts": end_ts.isoformat(),
            "initial_bankroll_cents": 100000,
            "fee_model_version": "kalshi_oct_2025",
            "execution_mode": "maker_taker",
            "observation_depth": -1,
            "has_trades_tape": len(self.trade_rows) > 0
                or (self.output_dir / "trades.parquet").exists(),
            "description": f"Live-recorded episode for {self.event_ticker}",
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Wrote metadata.json")


# ── Main loop ─────────────────────────────────────────────────────────────────


async def record_event(api: KalshiAPI, event_ticker: str, output_base: Path):
    """Record a single event from discovery through settlement."""
    output_dir = output_base / event_ticker
    recorder = EpisodeRecorder(api, event_ticker, output_dir)

    print(f"\n{'='*60}")
    print(f"Recording event: {event_ticker}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    await recorder.discover_markets()

    print(f"\n  Starting data collection (Ctrl+C to stop and save)...")
    print(f"  Orderbook poll: every 5s | Trade poll: every 10s\n")

    # Status printer
    async def print_status():
        while not recorder.shutting_down:
            await asyncio.sleep(30)
            if not recorder.shutting_down:
                print(
                    f"  [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                    f"Orderbooks: {recorder.sequence_id} | "
                    f"Trades: {len(recorder.trade_rows)} | "
                    f"Buffer: {len(recorder.orderbook_rows)} rows"
                )

    # Phase 1: Poll orderbooks + trades until close_time or Ctrl+C
    polling_tasks = [
        asyncio.create_task(recorder.poll_orderbooks()),
        asyncio.create_task(recorder.poll_trades()),
        asyncio.create_task(print_status()),
    ]

    # Wait for close_time in the main coroutine
    close_times = []
    for m in recorder.markets.values():
        ct = m.get("close_time") or m.get("expiration_time")
        if ct:
            try:
                close_times.append(datetime.fromisoformat(ct.replace("Z", "+00:00")))
            except ValueError:
                pass

    latest_close = max(close_times) if close_times else None

    user_interrupted = False
    try:
        if latest_close:
            now = datetime.now(timezone.utc)
            if now < latest_close:
                wait_secs = (latest_close - now).total_seconds()
                print(
                    f"  Markets close at {latest_close.isoformat()}, "
                    f"waiting {wait_secs/3600:.1f}h..."
                )
                while (
                    datetime.now(timezone.utc) < latest_close
                    and not recorder.shutting_down
                ):
                    await asyncio.sleep(5)
        else:
            # No close time — record until Ctrl+C
            print("  No close_time found, recording until Ctrl+C...")
            while not recorder.shutting_down:
                await asyncio.sleep(5)
    except (asyncio.CancelledError, KeyboardInterrupt):
        user_interrupted = True
    finally:
        # Stop polling
        recorder.shutting_down = True
        for t in polling_tasks:
            t.cancel()
        await asyncio.gather(*polling_tasks, return_exceptions=True)

    # Phase 2: Poll for settlement results (only if markets closed naturally)
    settlements = {}
    if latest_close and not user_interrupted:
        recorder.shutting_down = False
        settlements = await recorder.wait_for_settlement()

    # Phase 3: Write outputs
    print(f"\n  Writing output files...")
    recorder.write_outputs(settlements)
    print(f"\n  Done! Episode saved to {output_dir}")


async def discover_active_event(api: KalshiAPI, series_ticker: str) -> str | None:
    """Find the active event for a series."""
    markets = await api.get_markets(series_ticker=series_ticker, status="open")
    if not markets:
        # Try all statuses
        markets = await api.get_markets(series_ticker=series_ticker)

    # Group by event_ticker and find the one with most open markets
    events: dict[str, int] = {}
    for m in markets:
        et = m.get("event_ticker", "")
        if et:
            events[et] = events.get(et, 0) + 1

    if not events:
        return None

    # Return event with most markets (most likely the active one)
    best = max(events, key=events.get)
    print(f"  Series {series_ticker} → event {best} ({events[best]} markets)")
    return best


async def main():
    parser = argparse.ArgumentParser(
        description="Record live Kalshi market data into backtest episode format"
    )
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        help="Event ticker to record (e.g. KXHIGHNY-26MAR15)",
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Series ticker — auto-discovers active event (e.g. KXHIGHNY)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="episodes",
        help="Output directory (default: episodes/)",
    )
    args = parser.parse_args()

    if not args.event and not args.series:
        parser.error("Provide at least one --event or --series")

    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.getenv("KALSHI_API_KEY_ID")
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not api_key or not pk_path:
        print("ERROR: Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env")
        sys.exit(1)

    private_key = load_private_key(pk_path)
    api = KalshiAPI(api_key, private_key)
    output_base = Path(args.output)

    # On Windows, handle Ctrl+C via KeyboardInterrupt since
    # add_signal_handler is not supported
    try:
        # Resolve series to events
        event_tickers = list(args.event)
        for series in args.series:
            event = await discover_active_event(api, series)
            if event:
                event_tickers.append(event)
            else:
                print(f"  [WARN] No active event found for series {series}")

        if not event_tickers:
            print("ERROR: No events to record")
            sys.exit(1)

        # Record events (parallel if multiple)
        if len(event_tickers) == 1:
            await record_event(api, event_tickers[0], output_base)
        else:
            tasks = [
                record_event(api, et, output_base) for et in event_tickers
            ]
            await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        print("\n  Interrupted — partial data has been saved.")
    finally:
        await api.close()


if __name__ == "__main__":
    # Windows needs this for Ctrl+C to work with asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
