import json
from pathlib import Path

trades_path = Path("results_momentumsniper/trades.json")
trades = json.loads(trades_path.read_text()) if trades_path.exists() else []

print(f"Total trades: {len(trades)}")
if not trades:
    print("No trades found")
    exit()

# Print all trade keys to understand structure
print(f"\nTrade keys: {list(trades[0].keys())}")
print(f"\nFirst 5 trades:")
for t in trades[:5]:
    print(f"  {t}")

# Analyze by episode
episodes = {}
for t in trades:
    ep = t.get("episode_id", "unknown")
    if ep not in episodes:
        episodes[ep] = []
    episodes[ep].append(t)

print(f"\nTrades per episode:")
for ep, ep_trades in episodes.items():
    pnl = sum(t.get("pnl_cents", 0) for t in ep_trades)
    print(f"  {ep}: {len(ep_trades)} trades, PnL: {pnl/100:.2f}")
