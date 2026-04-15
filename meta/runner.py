"""
Meta runner: orchestrates the full ASI-Evolve style cycle.

Weekly cycle (runs Sunday night):
  1. Analyze any active experiments that have reached MIN_ACTIVE_DAYS
  2. Activate any proposed experiments (set status → active, record accuracy_before)
  3. If no pending experiments exist → call Researcher to propose a new one
  4. Run weight optimizer: nudge Kronos/Reddit/Technicals weights based on accuracy

Designed to be called from the APScheduler alongside the daily pipeline.
"""
import logging
from datetime import datetime, timezone

from meta.analyzer import analyze_experiment
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

        # Step 1: analyze any mature active experiments
        active = self.store.get_active_experiments()
        analyzed = 0
        for exp in active:
            if analyze_experiment(exp, self.store):
                analyzed += 1
        if analyzed:
            logger.info("meta: analyzed %d experiments", analyzed)

        # Step 2: activate any proposed experiments
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

        # Step 3: propose new source if nothing pending
        still_pending = self.store.get_pending_experiments()
        if not still_pending:
            logger.info("meta: no pending experiments, asking Researcher for new proposal...")
            proposal = propose(self.store)
            if proposal:
                exp_id = self.store.create_experiment(proposal)
                logger.info("meta: new experiment created (id=%d): %s", exp_id, proposal["source_name"])
            else:
                logger.warning("meta: Researcher returned no proposal")
        else:
            logger.info("meta: %d experiment(s) pending activation, skipping new proposal", len(still_pending))

        duration = (datetime.now(timezone.utc) - t0).total_seconds()
        logger.info("meta: weekly cycle complete in %.1fs", duration)


def _current_accuracy(store) -> tuple[float | None, int]:
    try:
        stats = store.get_accuracy_stats()
        if stats["total"] == 0:
            return None, 0
        return stats["accuracy"], stats["total"]
    except Exception:
        return None, 0
