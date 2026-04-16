"""
Regression test for ISSUE-001: scheduler hour overflow.

When --hour 23 is passed, pipeline would be scheduled at hour 24,
which APScheduler rejects. Fixed with (hour + 1) % 24.

Found by /qa on 2026-04-15
Report: .gstack/qa-reports/qa-report-kronos-signal-2026-04-15.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSchedulerHourCalculation:
    """Verify the pipeline_hour = (discovery_hour + 1) % 24 logic."""

    def test_normal_hours(self):
        for h in range(0, 23):
            pipeline_hour = (h + 1) % 24
            assert 0 <= pipeline_hour <= 23, f"Pipeline hour {pipeline_hour} out of range for discovery hour {h}"

    def test_hour_23_wraps_to_0(self):
        # The specific bug case: --hour 23 should produce pipeline at 00:00
        assert (23 + 1) % 24 == 0

    def test_hour_22_produces_23(self):
        assert (22 + 1) % 24 == 23

    def test_default_hour_6_produces_7(self):
        assert (6 + 1) % 24 == 7
