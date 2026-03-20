"""Rule-based hard gate for SignalFusion V2.

Blocks signal setups proven to lose money based on feature analysis.
Fully deterministic — no ML.
"""


class HardGate:
    """Instant-reject filter for proven losing setups."""

    def __init__(self, config: dict | None = None):
        self.config = config or self.default_config()
        self._block_count: dict[str, int] = {}

    @staticmethod
    def default_config() -> dict:
        """Gate config populated from feature extraction analysis.

        Key findings from 60K signals:
        - Bullish signals: 30% correct (vs 94% bearish) → block or heavily penalize
        - Sports: 48% correct → block
        - Tape-only signals: 45% correct → block
        - Composite 0.5+: 48% correct (worse than low composite!) → don't use as quality signal
        - PH signals: 100% correct → always pass
        """
        return {
            "min_spread_cents": 3,
            "min_hours_to_close": 0.5,
            "min_book_depth": 5,
            "blocked_categories": ["sports"],
            "block_bullish_non_ph": True,    # bullish without PH = 30% WR
            "block_tape_only": True,          # tape-only = 45% WR
            "always_pass_ph_only": True,      # PH signals = 100% WR
        }

    def should_block(self, features: dict) -> tuple[bool, str]:
        """Returns (blocked, reason). If blocked=True, skip this signal."""

        # PH-only signals always pass (100% correct in training data)
        if self.config.get("always_pass_ph_only"):
            if features.get("source_ph") and not features.get("source_momentum") and not features.get("source_tape"):
                return False, "ph_only_pass"

        # Category gate
        category = features.get("category", "")
        if category in self.config.get("blocked_categories", []):
            self._count("blocked_category")
            return True, f"blocked_category:{category}"

        # Bullish non-PH gate (30% win rate)
        if self.config.get("block_bullish_non_ph"):
            if features.get("direction") == "bullish" and not features.get("source_ph"):
                self._count("blocked_bullish")
                return True, "bullish_without_ph"

        # Tape-only gate (45% win rate)
        if self.config.get("block_tape_only"):
            if features.get("source_tape") and not features.get("source_ph") and not features.get("source_momentum"):
                self._count("blocked_tape_only")
                return True, "tape_only"

        # Spread gate
        if features.get("spread_cents", 0) < self.config.get("min_spread_cents", 3):
            self._count("blocked_spread")
            return True, f"spread_too_tight:{features.get('spread_cents')}c"

        # Time gate
        if features.get("hours_to_close", 999) < self.config.get("min_hours_to_close", 0.5):
            self._count("blocked_time")
            return True, f"too_close_to_resolution:{features.get('hours_to_close'):.1f}h"

        # Book depth gate
        total_depth = features.get("book_depth_bid", 0) + features.get("book_depth_ask", 0)
        if total_depth < self.config.get("min_book_depth", 5):
            self._count("blocked_depth")
            return True, f"thin_book:{total_depth}"

        return False, "passed"

    def _count(self, reason: str) -> None:
        self._block_count[reason] = self._block_count.get(reason, 0) + 1

    @property
    def block_stats(self) -> dict[str, int]:
        return dict(self._block_count)

    def reset_stats(self) -> None:
        self._block_count.clear()
