"""
Forward-test evaluator.

Queries signals from the DB, groups by signal_version, and reports:
- Hit rate per label tier (did STRONG_BUY actually go up next day?)
- Mean signed return per tier
- Sample sizes

Usage:
    python scripts/forward_test.py
    python scripts/forward_test.py --version abc123def456
    python scripts/forward_test.py --days 30
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datetime import date, timedelta

from sqlalchemy import select

from storage.db import get_session, init_db
from storage.models import Signal


def _hit(label: str, price_at: float, price_next: float) -> bool:
    if label in ("STRONG_BUY", "BUY"):
        return price_next > price_at
    if label in ("STRONG_SELL", "SELL"):
        return price_next < price_at
    return False


def main():
    parser = argparse.ArgumentParser(description="Forward-test signal evaluator")
    parser.add_argument("--version", help="Filter to a specific signal_version hash")
    parser.add_argument("--days", type=int, default=None, help="Limit to last N days")
    args = parser.parse_args()

    init_db()

    since_date = None
    if args.days:
        since_date = (date.today() - timedelta(days=args.days)).isoformat()

    with get_session() as session:
        q = select(Signal).where(
            Signal.price_at_signal.isnot(None),
            Signal.price_next_day.isnot(None),
        )
        if args.version:
            q = q.where(Signal.signal_version == args.version)
        if since_date:
            q = q.where(Signal.signal_date >= since_date)
        rows = session.execute(q).scalars().all()

    if not rows:
        print("No resolved signals found (price_next_day is NULL for all matching rows).")
        return

    # Group: (signal_version, classifier_label) → list of (price_at, price_next)
    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r.signal_version or "unknown", r.classifier_label or "UNKNOWN")
        groups.setdefault(key, []).append((r.price_at_signal, r.price_next_day))

    LABEL_ORDER = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "UNKNOWN"]

    def _label_rank(label):
        try:
            return LABEL_ORDER.index(label)
        except ValueError:
            return 99

    sorted_keys = sorted(groups, key=lambda k: (k[0], _label_rank(k[1])))

    # Print header
    print(f"\n{'Version':<14}  {'Label':<12}  {'N':>5}  {'Hit%':>7}  {'Mean Ret%':>10}  {'Note'}")
    print("-" * 68)

    low_n_warning = False
    for version, label in sorted_keys:
        pairs = groups[(version, label)]
        n = len(pairs)
        hits = sum(1 for p_at, p_next in pairs if _hit(label, p_at, p_next))
        returns = [(p_next - p_at) / p_at * 100 for p_at, p_next in pairs]
        mean_ret = sum(returns) / n

        hit_pct = hits / n * 100 if label not in ("HOLD", "UNKNOWN") else float("nan")
        note = ""
        if n < 30:
            note = "⚠ n<30 (not significant)"
            low_n_warning = True

        hit_str = f"{hit_pct:.1f}%" if hit_pct == hit_pct else "  —  "  # nan check
        print(f"{version:<14}  {label:<12}  {n:>5}  {hit_str:>7}  {mean_ret:>+9.2f}%  {note}")

    print()
    if low_n_warning:
        print("⚠  Some groups have fewer than 30 signals — results are not statistically significant yet.")
    print(f"Total resolved signals: {len(rows)}")


if __name__ == "__main__":
    main()
