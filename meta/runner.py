"""
Meta runner: orchestrates the full ASI-Evolve style cycle.

Weekly cycle (runs Sunday night):
  1. Re-check any rotating experiments that now have enough signals for a confident decision
  2. Analyze any active experiments that have reached MIN_ACTIVE_DAYS
  3. Activate any proposed experiments
  4. If no pending experiments → call Researcher to propose a new one

Experiment statuses:
  proposed  → waiting to be activated
  active    → running, being evaluated weekly
  rotating  → benched (early negative signal), data preserved, awaiting more signals
  completed → kept permanently (accuracy held or improved)
  archived  → disabled permanently (confident it hurt accuracy)
"""
import logging
from datetime import datetime, timezone

from meta.analyzer import analyze_experiment, recheck_rotating, _try_promote_shadow
from meta.researcher import propose
from storage.db import init_db
from storage.store import SignalStore

logger = logging.getLogger(__name__)


class MetaRunner:
    def __init__(self):
        self.store = SignalStore()
        init_db()

    def run(self):
        logger.info("meta: starting weekly cycle")
        t0 = datetime.now(timezone.utc)

        # Step 0: check shadow experiments for promotion / redundancy
        shadow = self.store.get_shadow_experiments()
        promoted = 0
        for exp in shadow:
            if _try_promote_shadow(exp, self.store):
                promoted += 1
        if promoted:
            logger.info("meta: made shadow decisions on %d experiment(s)", promoted)

        # Step 1: re-check rotating experiments — maybe they have enough data now
        rotating = self.store.get_rotating_experiments()
        rechecked = 0
        for exp in rotating:
            if recheck_rotating(exp, self.store):
                rechecked += 1
        if rechecked:
            logger.info("meta: made confident decisions on %d rotating experiment(s)", rechecked)

        # Step 2: analyze active experiments
        active = self.store.get_active_experiments()
        analyzed = 0
        for exp in active:
            if analyze_experiment(exp, self.store):
                analyzed += 1
        if analyzed:
            logger.info("meta: analyzed %d active experiment(s)", analyzed)

        # Step 3: activate proposed experiments
        pending = self.store.get_pending_experiments()
        for exp in pending:
            acc, n = _current_accuracy(self.store)
            self.store.update_experiment(exp.id, {
                "status": "active",
                "accuracy_before": acc,
                "n_signals_before": n,
                "activated_at": datetime.now(timezone.utc).isoformat(),
            })
            logger.info("meta: activated experiment '%s'", exp.source_name)

        # Step 4: propose new source if nothing pending
        still_pending = self.store.get_pending_experiments()
        if not still_pending:
            logger.info("meta: no pending experiments — asking Researcher for new proposal...")
            proposal = propose(self.store)
            if proposal:
                exp_id = self.store.create_experiment(proposal)
                logger.info("meta: new experiment created (id=%d): %s", exp_id, proposal["source_name"])
            else:
                logger.warning("meta: Researcher returned no proposal")
        else:
            logger.info(
                "meta: %d experiment(s) pending activation, skipping new proposal",
                len(still_pending),
            )

        duration = (datetime.now(timezone.utc) - t0).total_seconds()
        logger.info(
            "meta: weekly cycle complete in %.1fs (rotated=%d analyzed=%d)",
            duration, rechecked, analyzed,
        )


def _current_accuracy(store) -> tuple[float | None, int]:
    try:
        stats = store.get_accuracy_stats()
        if stats["total"] == 0:
            return None, 0
        return stats["accuracy"], stats["total"]
    except Exception:
        return None, 0
