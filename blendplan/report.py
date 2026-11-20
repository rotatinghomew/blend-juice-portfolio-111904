"""Supervisor-facing report: every number traces back to evidence.

The report is deliberately verbose about *why* a number is firm or only
provisional.  It also keeps the usage ledger and the output ledger as two
separate corrections: confirming a usage changes what the next order can
promise; a finished-output row corrects the estimate and explains leftovers.
Neither substitutes for the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping

from .data import Dataset, Job
from .estimates import LotForecast, PilotModels, ReconciliationRow
from .ledger import Ledger, build_ledger
from .planner import Plan, Planner

STATUS_LABEL = {
    "firm": "可承诺",
    "conditional": "条件接单",
    "short": "需确认减量",
    "critical": "不可承诺（越窗）",
    "missed": "已过窗口（需人工改期）",
}


@dataclass(frozen=True)
class LotCard:
    lot: str
    kind: str
    revision: int
    net_kg: float
    splittable: bool
    expires: str
    quality_status: str
    bags: tuple[dict[str, object], ...]
    brix: float
    acid: float
    acid_gap: float
    se_brix: float | None
    forecast: LotForecast
    free_kg: float
    dosed_kg: float
    reserved_kg: float
    pending_kg: float
    over_locked_kg: float
    opened: bool
    evidence_note: str
    requirement: str | None


@dataclass(frozen=True)
class JobCard:
    job: str
    grade: str
    window: str
    minimum_juice_kg: float
    last_slot: str
    allows_lower: bool
    status: str
    slot: str
    raw_kg: float
    juice_kg: float
    shortfall_kg: float
    brix_out: float
    acid_out: float
    lots: Mapping[str, float]
    provisional_lots: tuple[str, ...]
    scenario_failures: tuple[str, ...]
    notes: tuple[str, ...]
    follow_up: tuple[str, ...]
    downgrade: str


@dataclass(frozen=True)
class PlanCard:
    label: str
    as_of: str
    objective: float
    jobs: tuple[JobCard, ...]
    lot_use: Mapping[str, float]
    leftover: Mapping[str, float]
    slot_raw: Mapping[str, float]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CommitmentImpact:
    job: str
    stated_minimum_kg: float
    best_juice_kg: float
    dosed_kg: float
    gap_kg: float
    classification: str
    reason: str
    action: str


@dataclass(frozen=True)
class Report:
    as_of: str
    evidence_cutoff: str
    excluded_events: tuple[str, ...]
    pilot_model: dict[str, object]
    coverage: tuple[dict[str, object], ...]
    lots: tuple[LotCard, ...]
    firm_plan: PlanCard
    provisional_plan: PlanCard
    ledger: dict[str, object]
    output_reconciliation: tuple[dict[str, object], ...]
    historical_check: tuple[dict[str, object], ...]
    commitment_impact: tuple[CommitmentImpact, ...]
    slots: tuple[dict[str, object], ...]
    recipe_notes: tuple[dict[str, object], ...]
    change_log: tuple[str, ...] = ()
    excluded_commitments: tuple[dict[str, object], ...] = ()
    caveats: tuple[str, ...] = field(default_factory=tuple)
    evidence_requests: tuple[str, ...] = field(default_factory=tuple)


def _window(job: Job) -> str:
    return (
        f"{job.min_brix:.2f}–{job.max_brix:.2f} °Bx，"
        f"{job.min_acid:.2f}–{job.max_acid:.2f}% 酸"
    )


def _downgrade_note(job: Job) -> str:
    if job.customer_allows_lower:
        return "客户允许降档：若补验失败，可提出转级建议，但须客户书面确认后生效"
    return "客户不允许降档：补验失败只能减量或改期，不能自行转级"


def _follow_up(job: Job, card_notes: Iterable[str], provisional: Iterable[str], failures: Iterable[str]) -> tuple[str, ...]:
    asks: list[str] = []
    if list(provisional):
        asks.append(
            "对暂估批次补同袋三层的复测（每袋至少 2 次读数），或做本档试榨；"
            "补齐前只能条件接单"
        )
    if list(failures):
        asks.append(
            "按全批一致偏差重算配比，或增加一档成品复配/复测后再确认窗口"
        )
    return tuple(asks)


def build_report(
    dataset: Dataset,
    as_of: datetime,
    *,
    previous_as_of: datetime | None = None,
    closed_slots: Iterable[str] = (),
) -> Report:
    models = PilotModels(dataset, as_of)
    ledger = build_ledger(dataset, as_of)
    planner = Planner(dataset, models, ledger, as_of, closed_slots=closed_slots)
    firm = planner.solve(firm_only=True)
    try:
        provisional = planner.solve(firm_only=False)
    except RuntimeError:
        # Firm mode may admit a lot whose dosed mass must stay on the books
        # even though it is not fully covered; the provisional mode is a
        # superset, so this should not normally happen.
        provisional = firm

    lots = dataset.lots(as_of)
    cards = tuple(
        _lot_card(dataset, models, ledger, lots[lot_id], planner.forecasts[lot_id])
        for lot_id in sorted(lots)
    )
    firm_card = _plan_card("firm", firm, models)
    provisional_card = _plan_card("provisional", provisional, models)
    impacts = _commitment_impact(dataset, firm, provisional, ledger)
    outputs = _output_reconciliation(dataset, models, ledger, as_of)
    history = _historical_check(models.reconcile())
    recipes = _recipe_notes(models, planner)
    late = [c for c in dataset.commitments if c.known_at > as_of]
    excluded = tuple(
        f"{c.event} {c.state} {c.qty_kg:g} kg 于 {c.known_at.isoformat()}" for c in late
    )
    change_log = (
        _change_log(previous_as_of, _snapshot(dataset, previous_as_of), _snapshot(dataset, as_of))
        if previous_as_of
        else ()
    )
    requests = _evidence_requests(cards)
    caveats = _caveats(models, ())
    return Report(
        as_of=as_of.isoformat(),
        evidence_cutoff=as_of.isoformat(),
        excluded_events=excluded,
        excluded_commitments=tuple(
            {
                "event": c.event,
                "job": c.job,
                "state": c.state,
                "lot": c.lot,
                "qty_kg": c.qty_kg,
                "known_at": c.known_at.isoformat(),
                "reason": "证据截止之后才可见，未进入本次用量账",
            }
            for c in late
        ),
        pilot_model=_model_summary(models),
        coverage=tuple(c.described() for c in models._coverage.values()),
        lots=cards,
        firm_plan=firm_card,
        provisional_plan=provisional_card,
        ledger=_ledger_summary(ledger),
        output_reconciliation=outputs,
        historical_check=history,
        commitment_impact=impacts,
        slots=_slot_rows(dataset, planner, provisional, as_of),
        recipe_notes=recipes,
        change_log=change_log,
        caveats=caveats,
        evidence_requests=requests,
    )


# -- lot cards ---------------------------------------------------------
def _lot_card(
    dataset: Dataset,
    models: PilotModels,
    ledger: Ledger,
    lot,
    forecast: LotForecast,
) -> LotCard:
    usage = ledger.lot(lot.lot)
    bags = tuple(
        {
            "prep": bag.prep,
            "layer": bag.layer,
            "version": bag.version,
            "brix": round(bag.brix, 3),
            "acid": round(bag.acid_percent, 4),
            "replicates": bag.replicates,
            "superseded": [
                {"version": v, "brix": round(b, 3), "acid": round(a, 4)}
                for v, b, a, _ in bag.superseded
            ],
        }
        for bag in lot.quality.bags
    )
    requirement = None
    if forecast.status == "uncovered":
        requirement = (
            f"补 {lot.kind} 同等级来料试榨至少 3 批（覆盖本次糖度 "
            f"{lot.quality.brix - 0.5:.1f}–{lot.quality.brix + 0.5:.1f} °Bx），"
            "并按袋记录三层与复测版本，方可转为可承诺原料"
        )
    elif forecast.status == "extrapolated":
        requirement = (
            "补测同袋三层的酸度复测或做本档试榨，使酸-糖线偏差落入覆盖范围；"
            "在此之前成品酸度只能作条件承诺"
        )
    return LotCard(
        lot=lot.lot,
        kind=lot.kind,
        revision=lot.rev,
        net_kg=lot.net_kg,
        splittable=lot.splittable,
        expires=lot.expires.isoformat(),
        quality_status=forecast.status,
        bags=bags,
        brix=lot.quality.brix,
        acid=lot.quality.acid_percent,
        acid_gap=forecast.acid_gap,
        se_brix=lot.quality.se_brix,
        forecast=forecast,
        free_kg=usage.free_kg,
        dosed_kg=usage.dosed_kg,
        reserved_kg=usage.reserved_kg,
        pending_kg=usage.pending_kg,
        over_locked_kg=usage.over_locked_kg,
        opened=usage.opened,
        evidence_note="；".join(lot.quality.notes) if lot.quality.notes else "袋级读数一致",
        requirement=requirement,
    )


# -- plan cards --------------------------------------------------------
def _plan_card(label: str, plan: Plan, models: PilotModels) -> PlanCard:
    jobs = []
    for job_id in sorted(plan.jobs):
        job_plan = plan.jobs[job_id]
        job = models.dataset.job(job_id)
        provisional = tuple(sorted(job_plan.provisional_lots))
        jobs.append(
            JobCard(
                job=job_id,
                grade=job.grade,
                window=_window(job),
                minimum_juice_kg=job.minimum_juice_kg,
                last_slot=job.last_slot,
                allows_lower=job.customer_allows_lower,
                status=job_plan.status,
                slot=job_plan.slot,
                raw_kg=job_plan.raw_kg,
                juice_kg=job_plan.juice_kg,
                shortfall_kg=job_plan.shortfall_kg,
                brix_out=job_plan.brix_out,
                acid_out=job_plan.acid_out,
                lots=dict(sorted(job_plan.lots.items())),
                provisional_lots=provisional,
                scenario_failures=job_plan.scenario_failures,
                notes=tuple(job_plan.notes),
                follow_up=_follow_up(
                    job, job_plan.notes, provisional, job_plan.scenario_failures
                ),
                downgrade=_downgrade_note(job),
            )
        )
    return PlanCard(
        label=label,
        as_of=plan.as_of.isoformat(),
        objective=plan.objective,
        jobs=tuple(jobs),
        lot_use=dict(sorted(plan.lot_use.items())),
        leftover={
            lot_id: round(qty, 1)
            for lot_id, qty in sorted(plan.leftover_kg.items())
            if qty > 1e-6
        },
        slot_raw=dict(sorted(plan.slot_raw.items())),
        notes=tuple(plan.notes),
    )


# -- ledger / output ---------------------------------------------------
def _ledger_summary(ledger: Ledger) -> dict[str, object]:
    return {
        "as_of": ledger.as_of.isoformat(),
        "usage": [
            {
                "lot": usage.lot,
                "net_kg": usage.net_kg,
                "dosed_kg": usage.dosed_kg,
                "reserved_kg": usage.reserved_kg,
                "pending_kg": usage.pending_kg,
                "free_kg": usage.free_kg,
                "opened": usage.opened,
                "over_locked_kg": usage.over_locked_kg,
            }
            for usage in sorted(ledger.lots.values(), key=lambda u: u.lot)
        ],
        "outputs": [
            {
                "job": job,
                "finished_juice_kg": usage.finished_juice_kg,
            }
            for job, usage in sorted(ledger.jobs.items())
            if usage.finished_juice_kg
        ],
        "rule": (
            "锁料/待批是计划，只有已投料扣减可用量；成品入库单列，"
            "用于校正估计与解释剩料，不能当作投料或反之"
        ),
    }


def _output_reconciliation(
    dataset: Dataset,
    models: PilotModels,
    ledger: Ledger,
    as_of: datetime,
) -> tuple[dict[str, object], ...]:
    """Compare finished-juice rows with the *dosed* material only.

    Finished juice corrects the yield estimate for the material actually
    opened.  It does not confirm or cancel a lock, and leftover opened
    material keeps its original lot evidence.
    """

    rows = []
    for event in ledger.raw_events:
        if event.state != "成品入库":
            continue
        usage = ledger.jobs.get(event.job)
        dosed = dict(usage.dosed_by_lot) if usage else {}
        estimated_juice = 0.0
        dosed_mass = 0.0
        for lot_id, qty in dosed.items():
            forecast = models.forecast_lot(dataset.lot_state(lot_id, as_of))
            estimated_juice += qty * forecast.point.yield_
            dosed_mass += qty
        implied_yield = event.qty_kg / dosed_mass if dosed_mass else None
        rows.append(
            {
                "event": event.event,
                "job": event.job,
                "slot": event.slot,
                "known_at": event.known_at.isoformat(),
                "finished_juice_kg": event.qty_kg,
                "dosed_by_lot": dict(sorted(dosed.items())),
                "dosed_kg": round(dosed_mass, 1),
                "dosed_expected_juice_kg": round(estimated_juice, 1),
                "output_vs_dosed_estimate_kg": round(event.qty_kg - estimated_juice, 1),
                "implied_yield": round(implied_yield, 3) if implied_yield else None,
                "forecast_yield": round(
                    sum(
                        dosed[lot_id] * models.forecast_lot(dataset.lot_state(lot_id, as_of)).point.yield_
                        for lot_id in dosed
                    )
                    / dosed_mass,
                    3,
                )
                if dosed_mass
                else None,
                "meaning": (
                    "产出记录用于校正已投料部分的出汁率与核算剩料；"
                    "它不冲减投料，也不把已开袋余料当成新料"
                ),
            }
        )
    return tuple(rows)


# -- slots -------------------------------------------------------------
def _slot_rows(dataset: Dataset, planner: Planner, plan: Plan, as_of: datetime) -> tuple[dict[str, object], ...]:
    rows = []
    for slot in sorted(dataset.slots, key=lambda s: (s.key, s.line, s.slot)):
        used = plan.slot_raw.get(slot.slot, 0.0)
        job = next((j.job for j in plan.jobs.values() if j.slot == slot.slot), None)
        rows.append(
            {
                "slot": slot.slot,
                "line": slot.line,
                "day": slot.day.isoformat(),
                "period": slot.period,
                "max_raw_kg": slot.max_raw_kg,
                "planned_raw_kg": round(used, 1),
                "job": job,
                "cleanup_allowed": slot.cleanup_allowed,
                "closed": slot.slot in planner.closed_slots,
            }
        )
    return tuple(rows)


# -- commitments -------------------------------------------------------
def _commitment_impact(
    dataset: Dataset, firm: Plan, provisional: Plan, ledger: Ledger
) -> tuple[CommitmentImpact, ...]:
    impacts: list[CommitmentImpact] = []
    for job_id in dataset.job_ids:
        job = dataset.job(job_id)
        usage = ledger.jobs.get(job_id)
        if usage is None:
            continue
        outstanding = usage.reserved_by_lot or usage.pending_by_lot or usage.dosed_by_lot
        if not outstanding:
            continue
        best = provisional.jobs.get(job_id)
        firm_plan = firm.jobs.get(job_id)
        best_juice = best.juice_kg if best else 0.0
        firm_juice = firm_plan.juice_kg if firm_plan else 0.0
        best_status = best.status if best else "short"
        dosed_mass = sum(usage.dosed_by_lot.values())
        gap = max(job.minimum_juice_kg - best_juice, 0.0)
        firm_gap = max(job.minimum_juice_kg - firm_juice, 0.0)
        if best_status == "missed":
            classification = "需确认减量"
            reason = (
                f"该单窗口已过，已投料约 {dosed_mass:.0f} kg 不能退回另一单，"
                "未完成部分没有可靠替代"
            )
            action = "与客户确认减量或改期；不得自动转级"
        elif firm_gap <= 1e-6:
            classification = "可补救"
            reason = "仅用可承诺原料即可补足"
            action = "按可承诺方案锁定；补验后复核"
        elif gap <= 1e-6:
            classification = "可补救"
            reason = "需动用暂估批次或补验后才能补足"
            action = "先条件接单；补验/补样通过后再确认，失败则确认减量"
        elif usage.dosed_by_lot:
            classification = "需确认减量"
            reason = "已投料不能退回另一单，剩余原料不足以补足合同量"
            action = "与客户确认减量或改期；不得自动转级"
        else:
            classification = "没有可靠替代"
            reason = "剩余原料无本档试榨证据，暂估不足以支撑承诺"
            action = "补试榨/补样；在此之前只能条件接单或确认减量"
        impacts.append(
            CommitmentImpact(
                job=job_id,
                stated_minimum_kg=job.minimum_juice_kg,
                best_juice_kg=round(best_juice, 1),
                dosed_kg=round(dosed_mass, 1),
                gap_kg=round(gap, 1),
                classification=classification,
                reason=reason,
                action=action,
            )
        )
    return tuple(impacts)


# -- historical check --------------------------------------------------
def _historical_check(rows: list[ReconciliationRow]) -> tuple[dict[str, object], ...]:
    if not rows:
        return ()
    diagnostics = {
        "brix_rmse": (sum(r.brix_error**2 for r in rows) / len(rows)) ** 0.5,
        "acid_rmse": (sum(r.acid_error**2 for r in rows) / len(rows)) ** 0.5,
        "yield_rmse": (sum(r.yield_error**2 for r in rows) / len(rows)) ** 0.5,
    }
    table = [
        {
            "pilot": row.pilot,
            "kind": row.kind,
            "line": row.line,
            "evidence_at_start": row.evidence_at_start,
            "brix_in": round(row.brix_in, 3),
            "acid_in": round(row.acid_in, 4),
            "predicted_brix_out": round(row.predicted_brix_out, 3),
            "actual_brix_out": round(row.actual_brix_out, 3),
            "brix_error": round(row.brix_error, 3),
            "predicted_acid_out": round(row.predicted_acid_out, 4),
            "actual_acid_out": round(row.actual_acid_out, 4),
            "acid_error": round(row.acid_error, 4),
            "predicted_yield": round(row.predicted_yield, 3),
            "actual_yield": round(row.actual_yield, 3),
            "yield_error": round(row.yield_error, 3),
            "output_known_at": row.output_known_at.isoformat(),
        }
        for row in rows
    ]
    summary = {
        "n": len(rows),
        "brix_rmse": round(diagnostics["brix_rmse"], 3),
        "acid_rmse": round(diagnostics["acid_rmse"], 4),
        "yield_rmse": round(diagnostics["yield_rmse"], 3),
        "rule": (
            "复核只用开工前已知的读数（每袋最新版）；成品记录只用于事后核对，"
            "不回填开工前证据"
        ),
        "rows": table,
    }
    return (summary,)


# -- recipes -----------------------------------------------------------
def _recipe_notes(models: PilotModels, planner: Planner) -> tuple[dict[str, object], ...]:
    yield_fit = models.yield_fit
    acid_fit = models.acid_fit
    return (
        {
            "topic": "什么原料能提高兑现量",
            "finding": (
                f"出汁率随来料糖度上升（每 +1 °Bx 约 "
                f"{yield_fit.beta[1] * 100:.1f} 个百分点），"
                f"成品酸度随成品糖度下降（每 +1 °Bx 约 "
                f"{-acid_fit.beta[1]:.3f} 个百分点酸）。"
                "因此补高糖原料同时提高出汁与降低酸度；补低酸原料只解决酸度，"
                "不增加兑现量"
            ),
            "how_to_compare": (
                "按同一批料的一致糖酸读数比较，不把高读数、高出汁率、低损耗"
                "跨样本拼在一起"
            ),
        },
        {
            "topic": "价差是否来自设备",
            "finding": (
                "在控制来料种类与糖度后，产线（榨二）对出汁率的系数为 "
                f"{models.line_effect()['line2_beta'] * 100:+.1f} 个百分点"
                f"（t={models.line_effect()['line2_t']:.2f}），统计上不能与 0 区分。"
                "产线并非随机分配（水田几乎只在榨一），因此表面上的产线差异"
                "主要是来料构成，不是设备能力"
            ),
            "how_to_compare": (
                "比价时必须把种类构成和糖度水平对齐；没有价格字段，"
                "本报告不推断单价，只说明可比口径"
            ),
        },
        {
            "topic": "新配方/未覆盖来料",
            "finding": (
                "未覆盖来料按已知档最保守出汁率与合并糖-酸关系估算，"
                "区间明显加宽"
            ),
            "how_to_compare": (
                "需补至少 3 批同档试榨（覆盖本次糖度附近 ±0.5 °Bx）"
                "并按袋记录三层、复测版本，才可用于承诺"
            ),
        },
        {
            "topic": "每 100 kg 原料的边际兑现量（同一批料的一致读数）",
            "finding": {
                lot_id: {
                    "kind": view.lot.kind,
                    "juice_per_100kg": round(100 * view.forecast.point.yield_, 1),
                    "brix_out": round(view.forecast.point.brix_out, 2),
                    "acid_out": round(view.forecast.point.acid_out, 3),
                    "evidence": view.forecast.status,
                }
                for lot_id, view in planner.usable_lots(False).items()
            },
            "how_to_compare": (
                "额外加哪一种果就按这一行的同一批读数算；不能对同批料同时取"
                "最高出汁率、最低损耗再拼最低酸度"
            ),
        },
    )


# -- model summary -----------------------------------------------------
def _model_summary(models: PilotModels) -> dict[str, object]:
    return {
        "n_pilots": models.yield_fit.n,
        "acid_line": {
            "intercept": round(float(models.acid_line.beta[0]), 4),
            "slope": round(float(models.acid_line.beta[1]), 4),
            "resid_sd": round(models.acid_line.resid_sd, 4),
        },
        "yield": {
            "intercept": round(float(models.yield_fit.beta[0]), 4),
            "brix": round(float(models.yield_fit.beta[1]), 5),
            "pinggang": round(float(models.yield_fit.beta[2]), 5),
            "shuitian": round(float(models.yield_fit.beta[3]), 5),
            "resid_sd": round(models.yield_fit.resid_sd, 4),
        },
        "brix_out": {
            "intercept": round(float(models.brix_fit.beta[0]), 4),
            "brix": round(float(models.brix_fit.beta[1]), 4),
            "acid_gap": round(float(models.brix_fit.beta[2]), 4),
            "resid_sd": round(models.brix_fit.resid_sd, 4),
        },
        "acid_out": {
            "intercept": round(float(models.acid_fit.beta[0]), 4),
            "brix_out": round(float(models.acid_fit.beta[1]), 5),
            "resid_sd": round(models.acid_fit.resid_sd, 5),
        },
        "loss_note": (
            "loss_kg 按试榨单独记录，不与 usable_juice_kg 相加；"
            "各档损耗仅作过程参考"
        ),
        "loss_by_kind": {
            kind: {"mean": round(mean, 4), "sd": round(sd, 4)}
            for kind, (mean, sd) in models.loss_summary().items()
        },
    }


# -- change log --------------------------------------------------------
def _snapshot(dataset: Dataset, as_of: datetime | None):
    if as_of is None:
        return None
    models = PilotModels(dataset, as_of)
    ledger = build_ledger(dataset, as_of)
    planner = Planner(dataset, models, ledger, as_of)
    firm = planner.solve(firm_only=True)
    provisional = planner.solve(firm_only=False)
    return {
        "as_of": as_of,
        "models": models,
        "firm": firm,
        "provisional": provisional,
        "ledger": ledger,
        "lots": dataset.lots(as_of),
    }


def _change_log(previous_as_of, before, after) -> tuple[str, ...]:
    if before is None:
        return ()
    lines: list[str] = []
    before_lots = before["lots"]
    after_lots = after["lots"]
    for lot_id in sorted(set(before_lots) | set(after_lots)):
        old = before_lots.get(lot_id)
        new = after_lots.get(lot_id)
        if old and new:
            if abs(old.net_kg - new.net_kg) > 1e-6 or old.rev != new.rev:
                lines.append(
                    f"{lot_id} 过磅修正：rev{old.rev} {old.net_kg:.0f} kg → "
                    f"rev{new.rev} {new.net_kg:.0f} kg"
                )
            old_q = old.quality
            new_q = new.quality
            if any(b.was_retested for b in new_q.bags) and (
                abs(old_q.brix - new_q.brix) > 1e-6 or abs(old_q.acid_percent - new_q.acid_percent) > 1e-6
            ):
                lines.append(
                    f"{lot_id} 复测：糖度 {old_q.brix:.2f}→{new_q.brix:.2f}，"
                    f"酸度 {old_q.acid_percent:.3f}→{new_q.acid_percent:.3f}"
                    "（同袋新版本，不是新观测）"
                )
        elif new and not old:
            lines.append(f"{lot_id} 在本次快照前才可见（rev{new.rev}）")
    old_dosed = {
        (job, lot): qty
        for job, usage in before["ledger"].jobs.items()
        for lot, qty in usage.dosed_by_lot.items()
    }
    new_dosed = {
        (job, lot): qty
        for job, usage in after["ledger"].jobs.items()
        for lot, qty in usage.dosed_by_lot.items()
    }
    for key in sorted(set(old_dosed) | set(new_dosed)):
        if abs(old_dosed.get(key, 0.0) - new_dosed.get(key, 0.0)) > 1e-6:
            lines.append(
                f"{key[0]} 投料 {key[1]}：{old_dosed.get(key, 0.0):.0f} → "
                f"{new_dosed.get(key, 0.0):.0f} kg（已投料不可回退）"
            )
    old_outputs = {
        event.event: event
        for event in before["ledger"].raw_events
        if event.state == "成品入库"
    }
    new_outputs = {
        event.event: event
        for event in after["ledger"].raw_events
        if event.state == "成品入库"
    }
    for event_id in sorted(set(new_outputs) - set(old_outputs)):
        event = new_outputs[event_id]
        lines.append(
            f"{event.event} 成品入库 {event.qty_kg:.0f} kg（{event.job}）："
            "用于校正出汁率与核算剩料，不冲减投料"
        )
    old_firm = {job: plan for job, plan in before["firm"].jobs.items()}
    new_firm = {job: plan for job, plan in after["firm"].jobs.items()}
    for job_id in sorted(set(old_firm) | set(new_firm)):
        old_plan = old_firm.get(job_id)
        new_plan = new_firm.get(job_id)
        if old_plan is None or new_plan is None:
            lines.append(f"{job_id} 工单在两次快照间进出可承诺计划")
            continue
        changes = []
        if old_plan.slot != new_plan.slot:
            changes.append(f"槽 {old_plan.slot}→{new_plan.slot}")
        if abs(old_plan.juice_kg - new_plan.juice_kg) > 5:
            changes.append(
                f"成品汁 {old_plan.juice_kg:.0f}→{new_plan.juice_kg:.0f} kg"
            )
        if set(old_plan.lots) != set(new_plan.lots):
            changes.append(f"用料 {_mix(old_plan.lots)} → {_mix(new_plan.lots)}")
        if old_plan.status != new_plan.status:
            changes.append(
                f"状态 {STATUS_LABEL.get(old_plan.status, old_plan.status)}→"
                f"{STATUS_LABEL.get(new_plan.status, new_plan.status)}"
            )
        if changes:
            lines.append(f"{job_id} 工单需改：" + "；".join(changes))
    old_prov = {
        j: p for j, p in before["provisional"].jobs.items()
    }
    new_prov = {j: p for j, p in after["provisional"].jobs.items()}
    for job_id in sorted(set(old_prov) | set(new_prov)):
        old_plan = old_prov.get(job_id)
        new_plan = new_prov.get(job_id)
        if old_plan is None or new_plan is None:
            continue
        if set(old_plan.lots) != set(new_plan.lots) or (
            abs(old_plan.juice_kg - new_plan.juice_kg) > 5
        ):
            lines.append(
                f"{job_id} 待批工单重选料：" + _mix(old_plan.lots) + " → " + _mix(new_plan.lots)
            )

    new_jobs = set(after["firm"].jobs) | set(after["provisional"].jobs)
    empty = [
        job_id
        for job_id in sorted(new_jobs)
        if not after["provisional"].jobs[job_id].lots
        and not after["firm"].jobs[job_id].lots
    ]
    for job_id in empty:
        lines.append(
            f"{job_id} 没有可用原料，不能排产；需补料或与客户确认减量/改期"
        )
    if not lines:
        lines.append("两次快照之间工单无需调整")
    return tuple(lines)


def _mix(lots: Mapping[str, float]) -> str:
    return "、".join(f"{lot} {qty:.0f} kg" for lot, qty in sorted(lots.items()))


# -- caveats / requests ------------------------------------------------
def _evidence_requests(cards: Iterable[LotCard]) -> tuple[str, ...]:
    requests: list[str] = []
    for card in cards:
        if card.requirement:
            requests.append(f"{card.lot}（{card.kind}）：{card.requirement}")
    return tuple(requests)


def _caveats(models: PilotModels, opened_leftovers: Iterable[str]) -> tuple[str, ...]:
    notes = [
        "读数按袋（prep）聚合，同袋复测的新版本替换旧版本，不增加整车样本量",
        "已投料不可退回或转给另一单；受影响承诺按可补救/需确认减量/无替代分开处理",
        "锁定的成品窗口不自动后移；改期只能由人确认",
        "暂估原料只能用于条件接单，补验通过前不作承诺",
    ]
    return tuple(notes)
