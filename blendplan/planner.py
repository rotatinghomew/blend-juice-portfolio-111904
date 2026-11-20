"""Physical blending plan as a small MIP.

Modelled explicitly
-------------------
* one job owns one slot ("一槽一批"), and a slot hosts one job;
* a job whose finished window is already locked by a ``锁料``/``已投料``
  event keeps that slot -- it is never silently moved to a later order;
* ``max_raw_kg`` is the total raw mass in the slot, including material that
  was already dosed there;
* lots are limited to ``net_kg - 已投料``; a reservation (``锁料``) held by
  one job is not available to another job;
* ``splittable=0`` lots go to exactly one job in full (or not at all);
* a lot can only enter a slot on or before its ``expires`` date;
* a grade change on one line needs a slot with ``cleanup_allowed``;
* perishable lots are preferred (small expiry-rank cost), but that is a
  tie-break, not a promise that everything is used in time.

Quality uses one consistent forecast triple per lot.  Job windows are soft
with a high penalty, so an almost-feasible order is reported with its exact
exceedance instead of silently disappearing.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .data import Dataset, Lot, Slot
from .estimates import LotForecast, PilotModels
from .ledger import Ledger

SHORTFALL_COST = 1_000.0
# Quality is a contract condition alongside quantity: violating it is never
# cheaper than declaring a shortfall.  A plan only uses quality slack when no
# quality-feasible blend exists at all.
QUALITY_COST = 1_000_000.0
JUICE_REWARD = 1e-3
EXPIRY_COST = 1e-4


@dataclass(frozen=True)
class LotView:
    lot: Lot
    forecast: LotForecast
    free_kg: float
    expiry_rank: int
    locked_by: Mapping[str, float]
    allocatable_kg: float

    @property
    def locked_total(self) -> float:
        return sum(self.locked_by.values())


@dataclass
class JobPlan:
    job: str
    slot: str
    raw_kg: float
    juice_kg: float  # conservative (low-yield) commitment
    expected_juice_kg: float
    minimum_juice_kg: float
    shortfall_kg: float
    brix_out: float
    acid_out: float
    lots: dict[str, float]
    status: str = "firm"
    notes: list[str] = field(default_factory=list)
    brix_violation: float = 0.0
    acid_violation: float = 0.0
    scenario_failures: tuple[str, ...] = ()
    margin: dict[str, float] = field(default_factory=dict)
    provisional_lots: tuple[str, ...] = ()

    @property
    def meets_minimum(self) -> bool:
        return self.shortfall_kg <= 1e-6


@dataclass
class Plan:
    as_of: datetime
    firm_only: bool
    jobs: dict[str, JobPlan]
    lot_use: dict[str, float]
    leftover_kg: dict[str, float]
    slot_raw: dict[str, float]
    objective: float
    assignment: dict[str, str]
    dosed_use: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_juice(self) -> float:
        return sum(j.juice_kg for j in self.jobs.values())

    @property
    def shortfall(self) -> float:
        return sum(j.shortfall_kg for j in self.jobs.values())

    @property
    def conditions(self) -> list[str]:
        out: list[str] = []
        for job in self.jobs.values():
            out.extend(f"{job.job}: {note}" for note in job.notes)
        out.extend(self.notes)
        return out


def chronological(slots: Iterable[Slot]) -> list[Slot]:
    return sorted(slots, key=lambda s: (s.key, s.line, s.slot))


class Planner:
    def __init__(
        self,
        dataset: Dataset,
        models: PilotModels,
        ledger: Ledger,
        as_of: datetime,
        closed_slots: Iterable[str] = (),
    ) -> None:
        self.dataset = dataset
        self.models = models
        self.ledger = ledger
        self.as_of = as_of
        self.lots = dataset.lots(as_of)
        self.closed_slots = frozenset(closed_slots) | self._evidence_closed_slots()
        self.slots = tuple(
            s
            for s in dataset.slots
            if s.tank_available and s.slot not in self.closed_slots
        )
        self.forecasts = models.forecast_lots(self.lots)
        self._slot_by_id = {s.slot: s for s in self.slots}
        self._order = {s.slot: i for i, s in enumerate(chronological(self.slots))}
        self._dosed = self._dosed_by_job_lot()
        self._locked = self._locked_by_job_lot()
        self._fixed = self._fixed_windows()
        self._ranks = self._expiry_ranks()

    # -- setup ---------------------------------------------------------
    def _evidence_closed_slots(self) -> set[str]:
        """Slots already finished, so they can never be re-planned.

        A finished-juice row proves that slot ran.  A slot on an earlier day
        is also closed; a slot on the current day is *not* guessed closed --
        the supervisor declares it with ``--closed-slot`` because the source
        has no clock-level slot times.
        """

        closed: set[str] = set()
        for event in self.ledger.raw_events:
            if event.state == "成品入库" and event.slot:
                closed.add(event.slot)
        for slot in self.dataset.slots:
            if slot.day < self.as_of.date():
                closed.add(slot.slot)
                continue
            if slot.day > self.as_of.date():
                continue
            # Same day: only infer closure once the period has clearly ended.
            # 上午 -> after 12:00, 下午 -> after 18:00.  Anything finer needs
            # the supervisor's explicit ``--closed-slot``.
            hour = self.as_of.hour
            if slot.period == "上午" and hour >= 12:
                closed.add(slot.slot)
            elif slot.period == "下午" and hour >= 18:
                closed.add(slot.slot)
        return closed

    def _expiry_ranks(self) -> dict[str, int]:
        dates = sorted({lot.expires for lot in self.lots.values()})
        rank = {d: i for i, d in enumerate(dates)}
        return {lot_id: rank[lot.expires] for lot_id, lot in self.lots.items()}

    def _locked_by_job_lot(self) -> dict[tuple[str, str], float]:
        """Outstanding reservation per job/lot (gross lock minus its dosed)."""

        locked: dict[tuple[str, str], float] = {}
        for event in self.ledger.raw_events:
            if event.state != "锁料" or not event.lot:
                continue
            key = (event.job, event.lot)
            locked[key] = locked.get(key, 0.0) + event.qty_kg
        for (job_id, lot_id), dosed in self._dosed.items():
            key = (job_id, lot_id)
            if key in locked:
                locked[key] = max(locked[key] - dosed, 0.0)
        return {key: qty for key, qty in locked.items() if qty > 1e-9}

    def _dosed_by_job_lot(self) -> dict[tuple[str, str], float]:
        dosed: dict[tuple[str, str], float] = {}
        for job_id, usage in self.ledger.jobs.items():
            for lot_id, qty in usage.dosed_by_lot.items():
                dosed[(job_id, lot_id)] = dosed.get((job_id, lot_id), 0.0) + qty
        return dosed

    def _fixed_windows(self) -> dict[str, str]:
        """Jobs whose finished window is already locked to a slot."""

        fixed: dict[str, str] = {}
        for event in self.ledger.raw_events:
            if event.state not in ("锁料", "已投料") or not event.slot:
                continue
            if event.slot not in self._slot_by_id:
                continue
            previous = fixed.get(event.job)
            if previous and previous != event.slot:
                # conflicting locks: keep the earlier one, note it later
                continue
            fixed[event.job] = event.slot
        return fixed

    def usable_lots(self, firm_only: bool) -> dict[str, LotView]:
        """Lots the planner may draw on now.

        Already-dosed mass is always represented (it cannot be undone even if
        its lot is only provisional); in firm mode such a lot simply gets no
        *new* allocation.  Provisional lots with no dosed mass are omitted.
        """

        views: dict[str, LotView] = {}
        for lot_id, lot in self.lots.items():
            forecast = self.forecasts[lot_id]
            free = self.ledger.free_lot_kg(lot_id)
            locked_by: dict[str, float] = {}
            for (job_id, held_lot), qty in self._locked.items():
                if held_lot == lot_id:
                    locked_by[job_id] = locked_by.get(job_id, 0.0) + qty
            dosed_here = sum(
                qty for (job_id, held_lot), qty in self._dosed.items() if held_lot == lot_id
            )
            if free <= 1e-9 and dosed_here <= 1e-9:
                continue
            if firm_only and not forecast.firm_eligible and dosed_here <= 1e-9:
                continue
            allocatable = free
            if firm_only and not forecast.firm_eligible:
                allocatable = 0.0
            views[lot_id] = LotView(
                lot=lot,
                forecast=forecast,
                free_kg=free,
                expiry_rank=self._ranks[lot_id],
                locked_by=locked_by,
                allocatable_kg=allocatable,
            )
        return views

    # -- slot rules ----------------------------------------------------
    def _lot_allowed_in_slot(self, slot: Slot, lot: Lot) -> bool:
        return slot.day <= lot.expires

    def _assignment_ok(self, assignment: Mapping[str, str]) -> bool:
        for slot_id in assignment.values():
            if slot_id not in self._slot_by_id:
                return False
        if len(set(assignment.values())) != len(assignment):
            return False
        for job_id, slot_id in assignment.items():
            job = self.dataset.job(job_id)
            if job.last_slot in self._order and self._order[slot_id] > self._order[job.last_slot]:
                return False
        by_line: dict[str, list[Slot]] = {}
        for job_id, slot_id in assignment.items():
            by_line.setdefault(self._slot_by_id[slot_id].line, []).append(
                self._slot_by_id[slot_id]
            )
        for slots in by_line.values():
            ordered = chronological(slots)
            for earlier, later in zip(ordered, ordered[1:]):
                early_job = next(j for j, s in assignment.items() if s == earlier.slot)
                late_job = next(j for j, s in assignment.items() if s == later.slot)
                if (
                    self.dataset.job(early_job).grade != self.dataset.job(late_job).grade
                    and not later.cleanup_allowed
                ):
                    return False
        return True

    def closed_jobs(self) -> set[str]:
        """Jobs that may not be re-planned at all.

        * the contractual ``last_slot`` has already run; or
        * the slot their finished window was locked/dosed into has already
          run -- that window is never silently moved to a later order.
        """

        closed = set()
        for job in self.dataset.jobs:
            if job.last_slot not in self._slot_by_id:
                closed.add(job.job)
            fixed_slot = self._fixed.get(job.job)
            if fixed_slot and fixed_slot not in self._slot_by_id:
                closed.add(job.job)
        return closed

    def assignments(self) -> list[dict[str, str]]:
        jobs = [job for job in self.dataset.jobs if job.job not in self.closed_jobs()]
        fixed = {j: s for j, s in self._fixed.items() if s in self._slot_by_id}
        free_jobs = [j.job for j in jobs if j.job not in fixed]
        used = set(fixed.values())
        free_slots = [s.slot for s in self.slots if s.slot not in used]
        results: list[dict[str, str]] = []
        if len(free_jobs) > len(free_slots):
            return results
        for combo in itertools.permutations(free_slots, len(free_jobs)):
            assignment = dict(fixed)
            assignment.update(dict(zip(free_jobs, combo)))
            if self._assignment_ok(assignment):
                results.append(assignment)
        return results

    # -- solve ---------------------------------------------------------
    def solve(self, *, firm_only: bool) -> Plan:
        views = self.usable_lots(firm_only)
        if not views:
            raise RuntimeError(
                "本次快照没有可用原料（没有过磅记录或已全部投料）"
            )
        best: Plan | None = None
        for assignment in self.assignments():
            plan = self._solve_assignment(assignment, views, firm_only)
            if plan is None:
                continue
            if best is None or plan.objective < best.objective - 1e-9:
                best = plan
        if best is None:
            if not self.assignments():
                raise RuntimeError(
                    "没有满足最后槽位、清线和整批规则的槽位组合"
                )
            raise RuntimeError(
                "该快照下没有可行混配：批次到期日早于可用槽位，"
                "或整批/容量约束无法同时满足"
            )
        self._add_missed_jobs(best, views)
        return best

    def _add_missed_jobs(self, plan: Plan, views: Mapping[str, LotView]) -> None:
        """Report jobs whose locked window has already run; never move them.

        The source has no instruction to re-open a finished slot, so these
        jobs are listed with zero promise rather than silently pushed to a
        later order -- that decision belongs to the planner/salesperson.
        """

        missed = sorted(self.closed_jobs() - set(plan.jobs))
        for job_id in missed:
            job = self.dataset.job(job_id)
            lots: dict[str, float] = {}
            juice = 0.0
            sugar = 0.0
            acid = 0.0
            for (dosed_job, lot_id), qty in self._dosed.items():
                if dosed_job != job_id or lot_id not in views:
                    continue
                forecast = views[lot_id].forecast
                lots[lot_id] = lots.get(lot_id, 0.0) + qty
                juice += qty * forecast.point.yield_
                sugar += qty * forecast.point.yield_ * forecast.point.brix_out
                acid += qty * forecast.point.yield_ * forecast.point.acid_out
            plan.jobs[job_id] = JobPlan(
                job=job_id,
                slot=job.last_slot,
                raw_kg=sum(lots.values()),
                juice_kg=juice,
                expected_juice_kg=juice,
                minimum_juice_kg=job.minimum_juice_kg,
                shortfall_kg=max(job.minimum_juice_kg - juice, 0.0),
                brix_out=sugar / juice if juice else float("nan"),
                acid_out=acid / juice if juice else float("nan"),
                lots=lots,
                status="missed",
                notes=[
                    f"合同窗口 {job.last_slot} 已过或该槽已产出，不能自动挪到下一单；"
                    "已投料部分保留在账上，未完成量需人工确认改期或减量"
                ],
                provisional_lots=tuple(
                    sorted(
                        lot_id
                        for lot_id in lots
                        if not views[lot_id].forecast.firm_eligible
                    )
                ),
            )

    def _solve_assignment(
        self,
        assignment: Mapping[str, str],
        views: Mapping[str, LotView],
        firm_only: bool,
    ) -> Plan | None:
        closed = self.closed_jobs()
        jobs = [job for job in self.dataset.jobs if job.job not in closed]
        if not jobs:
            return None
        lot_ids = sorted(views)
        n_jobs, n_lots = len(jobs), len(lot_ids)
        n_x = n_jobs * n_lots
        n_short = n_jobs
        n_quality = 4 * n_jobs
        n_bin = sum(
            n_jobs for lid in lot_ids if not views[lid].lot.splittable
        )
        n_vars = n_x + n_short + n_quality + n_bin
        short_start = n_x
        quality_start = n_x + n_short
        bin_start = quality_start + n_quality

        def ix(j: int, l: int) -> int:
            return j * n_lots + l

        def qs(j: int, which: int) -> int:
            return quality_start + 4 * j + which

        cost = np.zeros(n_vars)
        for j, job in enumerate(jobs):
            # Earlier contractual windows are slightly more expensive to miss,
            # so a tie is broken in favour of the job that closes first.  The
            # gradient is small (1%) and never overrides quantity or quality.
            deadline_rank = self._order.get(job.last_slot, len(self._order))
            cost[short_start + j] = SHORTFALL_COST * (
                1.0 + 0.01 * (len(self._order) - deadline_rank)
            )
            for w in range(4):
                cost[qs(j, w)] = QUALITY_COST
        for j in range(n_jobs):
            for l, lot_id in enumerate(lot_ids):
                forecast = views[lot_id].forecast
                cost[ix(j, l)] = EXPIRY_COST * views[lot_id].expiry_rank
                cost[ix(j, l)] -= JUICE_REWARD * forecast.point.yield_
                # A provisional lot is usable, but only for conditional plans.
                if not forecast.firm_eligible and firm_only:
                    cost[ix(j, l)] += 10.0

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []

        def add(row: np.ndarray, lo: float, hi: float) -> None:
            rows.append(row)
            lower.append(lo)
            upper.append(hi)

        # lot availability and lock protection
        for l, lot_id in enumerate(lot_ids):
            view = views[lot_id]
            row = np.zeros(n_vars)
            for j in range(n_jobs):
                row[ix(j, l)] = 1.0
            add(row, 0.0, view.allocatable_kg)
            other_locks = view.locked_total
            # A reservation is not a pool.  When a lot's reservations fit the
            # weighed weight, each job keeps its own slice.  When they no
            # longer fit (e.g. the lot was re-weighed lower), forcing them
            # would make the plan infeasible; the conflict is reported and the
            # planner re-optimises allocation instead of pretending the
            # reservation still stands.
            for j, job in enumerate(jobs):
                held = view.locked_by.get(job.job, 0.0)
                room = view.allocatable_kg - (other_locks - held)
                if room < view.allocatable_kg - 1e-9:
                    row = np.zeros(n_vars)
                    row[ix(j, l)] = 1.0
                    add(row, 0.0, max(room, 0.0))


        # dosed material stays with its job.  ``x`` is the *additional* raw
        # mass allocated now; dosed mass is a constant added back when the
        # plan is read out, so it is not bounded here.

        # slot capacity, including dosed mass on that slot
        for slot_id in set(assignment.values()):
            slot = self._slot_by_id[slot_id]
            j = next(j for j, job in enumerate(jobs) if assignment[job.job] == slot_id)
            dosed_here = sum(
                qty
                for (job_id, _lot), qty in self._dosed.items()
                if job_id == jobs[j].job
            )
            row = np.zeros(n_vars)
            for l in range(n_lots):
                row[ix(j, l)] = 1.0
            add(row, 0.0, max(slot.max_raw_kg - dosed_here, 0.0))

        # expiry
        for j, job in enumerate(jobs):
            slot = self._slot_by_id[assignment[job.job]]
            for l, lot_id in enumerate(lot_ids):
                if not self._lot_allowed_in_slot(slot, views[lot_id].lot):
                    row = np.zeros(n_vars)
                    row[ix(j, l)] = 1.0
                    add(row, 0.0, 0.0)

        # whole lots: the entire lot goes to exactly one job, or not at all
        offset = bin_start
        bin_index: dict[str, int] = {}
        for lot_id in lot_ids:
            if views[lot_id].lot.splittable:
                continue
            l = lot_ids.index(lot_id)
            job_bins: list[int] = []
            for j in range(n_jobs):
                bin_index[f"{lot_id}@{j}"] = offset
                row = np.zeros(n_vars)
                row[ix(j, l)] = 1.0
                row[offset] = -views[lot_id].allocatable_kg
                add(row, 0.0, 0.0)
                job_bins.append(offset)
                offset += 1
            row = np.zeros(n_vars)
            for position in job_bins:
                row[position] = 1.0
            add(row, 0.0, 1.0)

        # juice, sugar, acid balance per job
        for j, job in enumerate(jobs):
            row_juice = np.zeros(n_vars)
            row_sugar = np.zeros(n_vars)
            row_acid = np.zeros(n_vars)
            fixed_juice = fixed_sugar = fixed_acid = 0.0
            for (job_id, lot_id), qty in self._dosed.items():
                if job_id != job.job or lot_id not in views:
                    continue
                forecast = views[lot_id].forecast
                fixed_juice += qty * forecast.low.yield_
                fixed_sugar += qty * forecast.low.yield_ * forecast.point.brix_out
                fixed_acid += qty * forecast.low.yield_ * forecast.point.acid_out
            for l, lot_id in enumerate(lot_ids):
                forecast = views[lot_id].forecast
                # Commit quantity on the conservative (low) yield: a contract
                # minimum is a promise, not an average.  Concentrations use
                # the same-bag point reading, so the triple stays consistent.
                juice = forecast.low.yield_
                row_juice[ix(j, l)] = juice
                row_sugar[ix(j, l)] = juice * forecast.point.brix_out
                row_acid[ix(j, l)] = juice * forecast.point.acid_out

            row = row_juice.copy()
            row[short_start + j] = 1.0
            add(row, job.minimum_juice_kg - fixed_juice, np.inf)

            # sugar/brix: sum U*B + slack >= min*U ; <= max*U + slack
            for which, coef, fixed_mass, bound, sense in (
                (0, row_sugar, fixed_sugar, job.min_brix, 1.0),
                (1, row_sugar, fixed_sugar, job.max_brix, -1.0),
                (2, row_acid, fixed_acid, job.min_acid, 1.0),
                (3, row_acid, fixed_acid, job.max_acid, -1.0),
            ):
                row = sense * coef - sense * bound * row_juice
                row[qs(j, which)] = -1.0
                add(row, sense * bound * fixed_juice - sense * fixed_mass, np.inf)

            # keep the plan meaningful: juice >= min - short already; no
            # additional cap needed.

        if not rows or n_vars == 0:
            return None
        matrix = np.array(rows)
        integrality = np.zeros(n_vars)
        for position in bin_index.values():
            integrality[position] = 1
        result = milp(
            cost,
            constraints=LinearConstraint(matrix, lower, upper),
            integrality=integrality,
            bounds=Bounds(np.zeros(n_vars), np.full(n_vars, np.inf)),
        )
        if not result.success:
            return None
        x = result.x

        job_plans: dict[str, JobPlan] = {}
        lot_use = {lot_id: 0.0 for lot_id in lot_ids}
        for j, job in enumerate(jobs):
            lots: dict[str, float] = {}
            raw = juice = expected_juice = sugar = acid = 0.0
            for (job_id, lot_id), qty in self._dosed.items():
                if job_id != job.job or lot_id not in views:
                    continue
                forecast = views[lot_id].forecast
                lots[lot_id] = lots.get(lot_id, 0.0) + qty
                raw += qty
                juice += qty * forecast.low.yield_
                expected_juice += qty * forecast.point.yield_
                sugar += qty * forecast.low.yield_ * forecast.point.brix_out
                acid += qty * forecast.low.yield_ * forecast.point.acid_out
            for l, lot_id in enumerate(lot_ids):
                qty = float(x[ix(j, l)])
                if qty <= 1e-6:
                    continue
                forecast = views[lot_id].forecast
                lots[lot_id] = lots.get(lot_id, 0.0) + qty
                lot_use[lot_id] += qty
                raw += qty
                juice += qty * forecast.low.yield_
                expected_juice += qty * forecast.point.yield_
                sugar += qty * forecast.low.yield_ * forecast.point.brix_out
                acid += qty * forecast.low.yield_ * forecast.point.acid_out
            brix = sugar / juice if juice > 0 else float("nan")
            acidity = acid / juice if juice > 0 else float("nan")
            job_plans[job.job] = JobPlan(
                job=job.job,
                slot=assignment[job.job],
                raw_kg=raw,
                juice_kg=juice,
                expected_juice_kg=expected_juice,
                minimum_juice_kg=job.minimum_juice_kg,
                shortfall_kg=float(x[short_start + j]),
                brix_out=brix,
                acid_out=acidity,
                lots=lots,
            )

        plan = Plan(
            as_of=self.as_of,
            firm_only=firm_only,
            jobs=job_plans,
            lot_use={
                k: v + sum(
                    qty
                    for (job_id, lot), qty in self._dosed.items()
                    if lot == k
                )
                for k, v in lot_use.items()
            },
            leftover_kg={
                lot_id: max(
                    view.free_kg
                    - sum(
                        float(x[ix(j, li)])
                        for j in range(n_jobs)
                        for li, candidate in enumerate(lot_ids)
                        if candidate == lot_id
                    ),
                    0.0,
                )
                for lot_id, view in views.items()
            },
            dosed_use={
                job_id: dict(usage.dosed_by_lot)
                for job_id, usage in self.ledger.jobs.items()
            },
            slot_raw={
                slot_id: sum(jp.raw_kg for jp in job_plans.values() if jp.slot == slot_id)
                for slot_id in set(assignment.values())
            },
            objective=float(result.fun),
            assignment=dict(assignment),
        )
        self._classify(plan, views, firm_only)
        return plan

    # -- classification ------------------------------------------------
    def _classify(
        self, plan: Plan, views: Mapping[str, LotView], firm_only: bool
    ) -> None:
        scenarios = self._global_scenarios(plan, views)
        for job_plan in plan.jobs.values():
            job = self.dataset.job(job_plan.job)
            notes: list[str] = []
            status = "firm"
            if job_plan.shortfall_kg > 1e-6:
                status = "short"
                notes.append(
                    f"预计成品汁 {job_plan.juice_kg:.0f} kg，低于合同最低量 "
                    f"{job.minimum_juice_kg:.0f} kg，缺 {job_plan.shortfall_kg:.0f} kg"
                )
            job_plan.brix_violation = max(
                0.0, job.min_brix - job_plan.brix_out, job_plan.brix_out - job.max_brix
            ) if job_plan.juice_kg else 0.0
            job_plan.acid_violation = max(
                0.0, job.min_acid - job_plan.acid_out, job_plan.acid_out - job.max_acid
            ) if job_plan.juice_kg else 0.0
            job_plan.margin = {
                "brix_low": job_plan.brix_out - job.min_brix,
                "brix_high": job.max_brix - job_plan.brix_out,
                "acid_low": job_plan.acid_out - job.min_acid,
                "acid_high": job.max_acid - job_plan.acid_out,
            }
            if job_plan.brix_violation > 1e-9 or job_plan.acid_violation > 1e-9:
                status = "critical"
                notes.append(
                    f"名义配比已越窗：糖度 {job_plan.brix_out:.2f}（窗口 "
                    f"{job.min_brix:.2f}–{job.max_brix:.2f}），酸度 "
                    f"{job_plan.acid_out:.3f}（窗口 {job.min_acid:.2f}–{job.max_acid:.2f}）"
                )
            provisional = [
                lot_id
                for lot_id in job_plan.lots
                if not views[lot_id].forecast.firm_eligible
            ]
            job_plan.provisional_lots = tuple(sorted(provisional))
            if provisional:
                if status == "firm":
                    status = "conditional"
                notes.append(
                    "依赖暂估原料 " + "、".join(sorted(provisional))
                    + "（证据不足以直接承诺）"
                )
            failures = scenarios.get(job_plan.job, {}) if job_plan.lots else {}
            job_plan.scenario_failures = tuple(
                name for name, ok in failures.items() if not ok
            )
            if job_plan.scenario_failures:
                if status == "firm":
                    status = "conditional"
                notes.append(
                    "全批一致情景复核未通过：" + "、".join(job_plan.scenario_failures)
                )
            if job_plan.brix_violation <= 1e-9 and job_plan.acid_violation <= 1e-9:
                tight = [
                    label
                    for label, value in job_plan.margin.items()
                    if abs(value) < 0.01
                ]
                if tight:
                    notes.append(
                        "质量余量很窄：" + "、".join(f"{k} {job_plan.margin[k]:+.3f}" for k in tight)
                    )
            if status in ("firm",) and not job_plan.meets_minimum:
                status = "short"
            job_plan.status = status
            job_plan.notes = notes

        if firm_only:
            plan.notes.append(
                "可承诺部分只使用试榨覆盖范围内的原料；已投料不可撤销部分单列"
            )
        else:
            plan.notes.append("包含暂估原料，结果只能用于条件接单")

    def _global_scenarios(
        self, plan: Plan, views: Mapping[str, LotView]
    ) -> dict[str, dict[str, bool]]:
        """Move every lot together, in both directions, across all jobs.

        A lot split between several jobs gets the *same* shift in each of
        them, so no job is allowed a "just good enough" draw that another job
        cannot reproduce.  Quantity is measured against the conservative
        plan (which already sizes on the low yield); concentrations are
        measured against the planned triple, so an upward yield move alone is
        not treated as a quality failure.
        """

        results: dict[str, dict[str, bool]] = {j.job: {} for j in plan.jobs.values()}
        used = {
            lot_id
            for job_plan in plan.jobs.values()
            for lot_id in job_plan.lots
            if lot_id in views
        }
        if not used:
            return results
        scenarios = ("低出汁率", "高出汁率", "高糖低酸", "低糖高酸")
        for name in scenarios:
            for job_plan in plan.jobs.values():
                job = self.dataset.job(job_plan.job)
                juice = sugar = acid = 0.0
                for lot_id, qty in job_plan.lots.items():
                    if lot_id not in views:
                        continue
                    forecast = views[lot_id].forecast
                    if name == "低出汁率":
                        y = forecast.low.yield_
                    elif name == "高出汁率":
                        y = forecast.high.yield_
                    else:
                        y = forecast.low.yield_
                    if name == "高糖低酸":
                        b, a = forecast.high.brix_out, forecast.high.acid_out
                    elif name == "低糖高酸":
                        b, a = forecast.low.brix_out, forecast.low.acid_out
                    else:
                        b, a = forecast.point.brix_out, forecast.point.acid_out
                    juice += qty * y
                    sugar += qty * y * b
                    acid += qty * y * a
                if juice <= 0:
                    results[job_plan.job][name] = False
                    continue
                brix = sugar / juice
                acidity = acid / juice
                if name == "低出汁率":
                    # The plan already sizes on this case, so a failure here
                    # means the promise itself is not covered.
                    results[job_plan.job][name] = (
                        juice >= job.minimum_juice_kg - 1e-6
                    )
                elif name == "高出汁率":
                    # More juice at the same blend only helps quantity; it can
                    # dilute concentration, so check the window at this yield.
                    results[job_plan.job][name] = (
                        juice >= job.minimum_juice_kg - 1e-6
                        and job.min_brix - 1e-9 <= brix <= job.max_brix + 1e-9
                        and job.min_acid - 1e-9 <= acidity <= job.max_acid + 1e-9
                    )
                else:
                    results[job_plan.job][name] = (
                        job.min_brix - 1e-9 <= brix <= job.max_brix + 1e-9
                        and job.min_acid - 1e-9 <= acidity <= job.max_acid + 1e-9
                    )
        return results


def solve_firm_plan(
    dataset: Dataset, models: PilotModels, ledger: Ledger, as_of: datetime
) -> Plan:
    return Planner(dataset, models, ledger, as_of).solve(firm_only=True)


def solve_provisional_plan(
    dataset: Dataset, models: PilotModels, ledger: Ledger, as_of: datetime
) -> Plan:
    return Planner(dataset, models, ledger, as_of).solve(firm_only=False)
