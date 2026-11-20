"""Invariant tests for the pre-blend analysis.

Each test pins one of the business rules that the source data is designed to
trip: bag-level evidence, version replacement, point-in-time visibility,
whole-lot constraints, ledger separation, and coupled quality scenarios.
"""

import unittest
from datetime import datetime
from pathlib import Path

from blendplan.data import Dataset
from blendplan.estimates import PilotModels
from blendplan.ledger import build_ledger
from blendplan.planner import Planner
from blendplan.report import build_report
from blendplan.render import render_markdown

ROOT = Path(__file__).resolve().parents[1]
SNAP_10 = datetime.fromisoformat("2026-11-10T09:00:00+08:00")
BEFORE_DOSE = datetime.fromisoformat("2026-11-11T07:20:00+08:00")
AFTER_DOSE = datetime.fromisoformat("2026-11-11T08:30:00+08:00")
AFTER_REWEIGH_BEFORE = datetime.fromisoformat("2026-11-11T07:59:00+08:00")
AFTER_OUTPUT = datetime.fromisoformat("2026-11-11T17:30:00+08:00")


def dataset():
    return Dataset.load(ROOT)


class BagEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.ds = dataset()

    def test_replicates_are_not_independent_trucks(self):
        quality = self.ds.quality("D1", SNAP_10)
        self.assertEqual(quality.n_bags, 3)
        self.assertEqual(len([r for r in self.ds.readings if r.sample == "D1"]), 6)

    def test_newer_version_replaces_same_bag(self):
        early = self.ds.quality("D2", SNAP_10)
        late = self.ds.quality("D2", AFTER_DOSE)
        # Before the re-test: three bags at the original values.
        self.assertEqual(early.n_bags, 3)
        self.assertFalse(early.partial_retest)
        # After it: still three bags, not four or five.
        self.assertEqual(late.n_bags, 3)
        self.assertTrue(late.partial_retest)
        bag = next(b for b in late.bags if b.prep == "D2-袋1")
        self.assertEqual(bag.version, 2)
        self.assertEqual(bag.replicates, 2)
        self.assertAlmostEqual(bag.brix, 10.15)
        self.assertTrue(bag.was_retested)

    def test_retest_is_not_a_second_truck(self):
        early = self.ds.quality("D2", SNAP_10)
        late = self.ds.quality("D2", AFTER_DOSE)
        self.assertNotAlmostEqual(early.brix, late.brix)
        # The second version pulls the truck level down; it does not add a
        # high observation that could be cherry-picked.
        self.assertLess(late.brix, early.brix)
        self.assertGreaterEqual(late.scenario_brix_low, 10.15 - 1e-9)

    def test_high_reading_does_not_become_whole_truck(self):
        quality = self.ds.quality("D5", SNAP_10)
        bag_levels = [b.brix for b in quality.bags]
        self.assertLessEqual(quality.brix, max(bag_levels))
        self.assertLess(quality.scenario_brix_low, quality.brix)


class VisibilityTest(unittest.TestCase):
    def setUp(self):
        self.ds = dataset()

    def test_reweigh_not_visible_before_known_at(self):
        before = self.ds.lot_state("D2", datetime.fromisoformat("2026-11-11T07:59:00+08:00"))
        after = self.ds.lot_state("D2", datetime.fromisoformat("2026-11-11T08:00:00+08:00"))
        self.assertEqual(before.rev, 1)
        self.assertEqual(before.net_kg, 410)
        self.assertEqual(after.rev, 2)
        self.assertEqual(after.net_kg, 292)

    def test_output_not_visible_before_known_at(self):
        early = build_ledger(self.ds, AFTER_DOSE)
        late = build_ledger(self.ds, AFTER_OUTPUT)
        self.assertEqual(early.job("J1").finished_juice_kg, 0)
        self.assertEqual(late.job("J1").finished_juice_kg, 111)

    def test_dosing_not_visible_before_known_at(self):
        before = build_ledger(self.ds, BEFORE_DOSE)
        after = build_ledger(self.ds, AFTER_DOSE)
        self.assertEqual(before.lot("D1").dosed_kg, 0)
        self.assertEqual(after.lot("D1").dosed_kg, 220)
        self.assertEqual(after.lot("D1").free_kg, 360)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.ds = dataset()

    def test_lock_is_not_consumption(self):
        ledger = build_ledger(self.ds, SNAP_10)
        self.assertEqual(ledger.lot("D1").reserved_kg, 240)
        self.assertEqual(ledger.lot("D1").dosed_kg, 0)
        self.assertEqual(ledger.lot("D1").free_kg, 580)

    def test_dosed_does_not_double_count_reservation(self):
        ledger = build_ledger(self.ds, AFTER_DOSE)
        d1 = ledger.lot("D1")
        # 240 locked, 220 of which has been dosed -> 20 still reserved.
        self.assertEqual(d1.dosed_kg, 220)
        self.assertEqual(d1.reserved_kg, 20)
        self.assertEqual(d1.free_kg, 360)

    def test_over_locked_lot_is_reported(self):
        ledger = build_ledger(self.ds, AFTER_DOSE)
        self.assertAlmostEqual(ledger.lot("D2").over_locked_kg, 118)

    def test_output_is_a_separate_ledger(self):
        ledger = build_ledger(self.ds, AFTER_OUTPUT)
        # Output never reduces dosed mass or frees a lock.
        self.assertEqual(ledger.lot("D1").dosed_kg, 220)
        self.assertEqual(ledger.job("J1").finished_juice_kg, 111)
        self.assertEqual(ledger.job("J1").dosed_by_lot["D1"], 220)

    def test_pending_work_order_keeps_its_own_lot(self):
        ledger = build_ledger(self.ds, AFTER_DOSE)
        self.assertEqual(ledger.lot("D4").pending_kg, 300)
        self.assertEqual(ledger.job("J2").pending_by_lot["D4"], 300)


class EstimateTest(unittest.TestCase):
    def setUp(self):
        self.ds = dataset()
        self.models = PilotModels(self.ds, SNAP_10)

    def test_acid_follows_brix(self):
        beta = self.models.acid_fit.beta
        self.assertLess(beta[1], 0)
        # finished acid is not freely chosen: it is a function of sugar
        high = self.models.acid_out(13.0)
        low = self.models.acid_out(11.0)
        self.assertLess(high, low)

    def test_scenarios_use_one_bag(self):
        lot = self.ds.lot_state("D5", SNAP_10)
        forecast = self.models.forecast_lot(lot)
        # Low scenario = low brix bag -> low yield, low finished brix, high acid
        self.assertLess(forecast.low.yield_, forecast.point.yield_)
        self.assertLess(forecast.low.brix_out, forecast.point.brix_out)
        self.assertGreater(forecast.low.acid_out, forecast.point.acid_out)
        # High scenario is the mirror image
        self.assertGreater(forecast.high.yield_, forecast.point.yield_)
        self.assertGreater(forecast.high.brix_out, forecast.point.brix_out)
        self.assertLess(forecast.high.acid_out, forecast.point.acid_out)

    def test_loss_never_added_to_juice(self):
        for pilot in self.ds.pilots[:5]:
            output = self.ds.pilot_output(pilot.pilot)
            self.assertLess(output.usable_juice_kg + output.loss_kg, pilot.raw_kg)

    def test_uncovered_kind_is_provisional(self):
        lot = self.ds.lot_state("D5", SNAP_10)
        forecast = self.models.forecast_lot(lot)
        self.assertEqual(forecast.status, "uncovered")
        self.assertFalse(forecast.firm_eligible)

    def test_covered_and_extrapolated_are_distinguished(self):
        self.assertTrue(self.models.forecast_lot(self.ds.lot_state("D3", SNAP_10)).firm_eligible)
        d1 = self.models.forecast_lot(self.ds.lot_state("D1", SNAP_10))
        self.assertEqual(d1.status, "extrapolated")
        self.assertFalse(d1.firm_eligible)

    def test_line_effect_controls_for_feed(self):
        effect = self.models.line_effect()
        # 水田 is almost entirely on 榨一, so the raw line comparison is
        # confounded; the controlled coefficient must not be significant.
        self.assertLess(abs(effect["line2_t"]), 2.0)
        self.assertGreater(effect["counts"][("水田", "榨一")], effect["counts"][("水田", "榨二")])

    def test_historical_check_uses_prestart_evidence(self):
        rows = self.models.reconcile()
        self.assertEqual(len(rows), 108)
        for row in rows:
            self.assertLessEqual(
                datetime.fromisoformat(row.evidence_at_start.split("最后读数 ")[1]),
                row.started_at,
            )
            self.assertGreater(row.output_known_at, row.started_at)


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.ds = dataset()

    def _planner(self, as_of):
        models = PilotModels(self.ds, as_of)
        ledger = build_ledger(self.ds, as_of)
        return Planner(self.ds, models, ledger, as_of)

    def test_whole_lot_never_split(self):
        planner = self._planner(SNAP_10)
        plan = planner.solve(firm_only=False)
        d3 = plan.jobs["J1"].lots.get("D3", 0) + plan.jobs["J3"].lots.get("D3", 0)
        holders = [job for job in plan.jobs.values() if "D3" in job.lots]
        self.assertEqual(len(holders), 1)
        self.assertAlmostEqual(d3, 390)

    def test_slot_capacity_respected(self):
        planner = self._planner(SNAP_10)
        plan = planner.solve(firm_only=False)
        for slot_id, raw in plan.slot_raw.items():
            self.assertLessEqual(raw, planner._slot_by_id[slot_id].max_raw_kg + 1e-6)

    def test_lot_limit_respects_dosed_mass(self):
        planner = self._planner(AFTER_DOSE)
        plan = planner.solve(firm_only=False)
        used = plan.lot_use
        self.assertLessEqual(used.get("D1", 0), 580 + 1e-6)
        self.assertLessEqual(used.get("D2", 0), 292 + 1e-6)

    def test_expiry_blocks_late_slot(self):
        planner = self._planner(SNAP_10)
        # D2 expires 2026-11-12; S4 is on 2026-11-13.
        for assignment in planner.assignments():
            if "D2" in planner.usable_lots(False):
                plan = planner._solve_assignment(
                    assignment, planner.usable_lots(False), firm_only=False
                )
                if plan is None:
                    continue
                for job in plan.jobs.values():
                    if "D2" in job.lots:
                        self.assertNotEqual(job.slot, "S4")

    def test_cleanup_rule_blocks_grade_switch(self):
        planner = self._planner(SNAP_10)
        # S2 is 上午 on 榨一 with cleanup_allowed=0 as a *final* slot only if a
        # grade switch happens; assignments that do so must be dropped.
        for assignment in planner.assignments():
            slots = {
                slot_id: planner._slot_by_id[slot_id]
                for slot_id in assignment.values()
            }
            for line in {s.line for s in slots.values()}:
                ordered = sorted(
                    [s for s in slots.values() if s.line == line], key=lambda s: s.key
                )
                for earlier, later in zip(ordered, ordered[1:]):
                    early_job = next(j for j, s in assignment.items() if s == earlier.slot)
                    late_job = next(j for j, s in assignment.items() if s == later.slot)
                    if (
                        self.ds.job(early_job).grade
                        != self.ds.job(late_job).grade
                    ):
                        self.assertTrue(later.cleanup_allowed)

    def test_locked_window_never_moves(self):
        planner = self._planner(AFTER_DOSE)
        plan = planner.solve(firm_only=False)
        # J1 was locked/dosed in S1 and must stay in S1.
        self.assertEqual(plan.jobs["J1"].slot, "S1")

    def test_firm_plan_uses_only_covered_lots(self):
        planner = self._planner(SNAP_10)
        plan = planner.solve(firm_only=True)
        provisional = {"D1", "D2", "D5"}
        for job in plan.jobs.values():
            self.assertFalse(provisional & set(job.lots))
            self.assertEqual(job.provisional_lots, ())

    def test_provisional_plan_is_conditional(self):
        planner = self._planner(SNAP_10)
        plan = planner.solve(firm_only=False)
        statuses = {job.status for job in plan.jobs.values()}
        self.assertIn("conditional", statuses)

    def test_last_slot_respected(self):
        planner = self._planner(SNAP_10)
        order = planner._order
        for assignment in planner.assignments():
            for job_id, slot_id in assignment.items():
                job = self.ds.job(job_id)
                self.assertLessEqual(order[slot_id], order[job.last_slot])


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.ds = dataset()

    def test_markdown_renders(self):
        report = build_report(self.ds, SNAP_10)
        text = render_markdown(report)
        self.assertIn("混配前分析", text)
        self.assertIn("可承诺方案", text)
        self.assertIn("D5", text)

    def test_commitment_impact_covers_open_jobs(self):
        report = build_report(self.ds, AFTER_DOSE)
        jobs = {impact.job for impact in report.commitment_impact}
        self.assertEqual(jobs, {"J1", "J2"})
        for impact in report.commitment_impact:
            self.assertIn(
                impact.classification, {"可补救", "需确认减量", "没有可靠替代"}
            )

    def test_change_log_names_affected_work_orders(self):
        report = build_report(self.ds, AFTER_DOSE, previous_as_of=SNAP_10)
        text = "\n".join(report.change_log)
        self.assertIn("D2", text)
        self.assertIn("rev1", text)
        self.assertIn("J1", text)
        self.assertIn("已投料不可回退", text)

    def test_output_correction_does_not_cancel_lock(self):
        report = build_report(self.ds, AFTER_OUTPUT)
        self.assertEqual(report.output_reconciliation[0]["finished_juice_kg"], 111)
        self.assertEqual(report.output_reconciliation[0]["dosed_kg"], 220)
        self.assertEqual(report.ledger["usage"][0]["dosed_kg"], 220)

    def test_future_output_excluded(self):
        report = build_report(self.ds, SNAP_10)
        self.assertTrue(any("R6" in event for event in report.excluded_events))
        self.assertEqual(report.output_reconciliation, ())

    def test_leftovers_are_reported_and_keep_lot_evidence(self):
        report = build_report(self.ds, AFTER_DOSE)
        self.assertEqual(report.firm_plan.leftover, {"D1": 360.0})
        card = next(c for c in report.lots if c.lot == "D1")
        self.assertTrue(card.opened)
        self.assertFalse(card.forecast.firm_eligible)

    def test_slots_section_marks_closed_slots(self):
        report = build_report(self.ds, AFTER_OUTPUT)
        closed = {row["slot"] for row in report.slots if row["closed"]}
        self.assertIn("S1", closed)

    def test_provisional_lots_have_evidence_requests(self):
        report = build_report(self.ds, SNAP_10)
        lots = {card.lot for card in report.lots if card.requirement}
        self.assertIn("D5", lots)
        self.assertIn("D1", lots)


class SourceFilesTest(unittest.TestCase):
    def test_no_files_written(self):
        before = {
            path: path.stat().st_mtime_ns
            for path in (ROOT / "data").glob("*.csv")
        }
        build_report(dataset(), SNAP_10)
        after = {
            path: path.stat().st_mtime_ns
            for path in (ROOT / "data").glob("*.csv")
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
