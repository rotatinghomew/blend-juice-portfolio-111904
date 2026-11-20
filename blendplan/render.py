"""Markdown rendering of :class:`blendplan.report.Report`."""

from __future__ import annotations

from .report import Report, STATUS_LABEL


def _kg(value: float) -> str:
    return f"{value:,.0f}"


def _mix(lots: dict[str, float]) -> str:
    return "、".join(f"{lot} {qty:.0f} kg" for lot, qty in sorted(lots.items()))


def render_markdown(report: Report) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 混配前分析")
    add("")
    add(f"- 证据截止：{report.evidence_cutoff}")
    add("- 口径：读数按袋聚合；换版只替换同袋旧值；已投料不回退；成品入库单列")
    add("")

    add("## 1. 原料是否有条件支撑订单")
    add("")
    add("| 批次 | 种类 | 净重 kg | 袋数 | 糖度 °Bx | 酸度 % | 酸-糖线偏差 | 证据状态 | 可用 kg |")
    add("|---|---|---:|---:|---:|---:|---:|---|---:|")
    for card in report.lots:
        add(
            f"| {card.lot} | {card.kind} | {_kg(card.net_kg)} | {len(card.bags)} | "
            f"{card.brix:.2f} | {card.acid:.3f} | {card.acid_gap:+.3f} | "
            f"{_status_text(card.quality_status)} | {_kg(card.free_kg)} |"
        )
    add("")
    for card in report.lots:
        label = {
            "covered": "试榨覆盖",
            "extrapolated": "超出覆盖（外推）",
            "uncovered": "无试榨（暂估）",
        }[card.quality_status]
        detail = (
            f"**{card.lot}** rev{card.revision}，{card.kind}，"
            f"{label}：{card.forecast.coverage_note}；"
            f"情景：{card.forecast.scenario_source}。"
            f"袋级读数：{card.evidence_note}。"
        )
        add(f"- {detail}")
        if card.over_locked_kg > 0:
            add(
                f"  - 注意：锁料/待批合计超出过磅净重 {_kg(card.over_locked_kg)} kg，"
                "工单不能再按原锁定量承诺。"
            )
        if card.requirement:
            add(f"  - 补证要求：{card.requirement}")
    add("")

    add("## 2. 可承诺方案（新增用料只用试榨覆盖内原料）")
    add("")
    _render_plan(add, report, report.firm_plan)
    add("")

    add("## 3. 条件接单方案（含暂估原料）")
    add("")
    _render_plan(add, report, report.provisional_plan)
    add("")

    add("## 4. 班次与槽位")
    add("")
    add("| 槽 | 线 | 日 | 时段 | 容量 kg | 本方案用量 kg | 工单 | 可清线 | 状态 |")
    add("|---|---|---|---|---:|---:|---|---|---|")
    for slot in report.slots:
        add(
            f"| {slot['slot']} | {slot['line']} | {slot['day']} | {slot['period']} | "
            f"{_kg(slot['max_raw_kg'])} | {_kg(slot['planned_raw_kg'])} | "
            f"{slot['job'] or '—'} | {'是' if slot['cleanup_allowed'] else '否'} | "
            f"{'已关闭' if slot['closed'] else '可用'} |"
        )
    add("")
    add("一槽一批；换档必须落在允许清线的时段；已过或已产出的槽不再排产。")
    add("")

    add("## 5. 承诺影响")
    add("")
    if report.commitment_impact:
        add(
            "| 工单 | 合同最低 kg | 条件方案可兑现 kg | 其中已投料 kg | 缺口 kg | 分类 | 说明与动作 |"
        )
        add("|---|---:|---:|---:|---:|---|---|")
        for impact in report.commitment_impact:
            add(
                f"| {impact.job} | {_kg(impact.stated_minimum_kg)} | "
                f"{_kg(impact.best_juice_kg)} | {_kg(impact.dosed_kg)} | "
                f"{_kg(impact.gap_kg)} | {impact.classification} | "
                f"{impact.reason}；{impact.action} |"
            )
    else:
        add("本次快照没有未结的锁料/待批/已投料事件。")
    add("")

    add("## 6. 用量账（锁料/待批/已投料）")
    add("")
    add("| 批次 | 净重 kg | 已投料 kg | 未投锁料 kg | 待批 kg | 可用 kg | 已开袋 | 超额锁料 kg |")
    add("|---|---:|---:|---:|---:|---:|---|---:|")
    for row in report.ledger["usage"]:
        add(
            f"| {row['lot']} | {_kg(row['net_kg'])} | {_kg(row['dosed_kg'])} | "
            f"{_kg(row['reserved_kg'])} | {_kg(row['pending_kg'])} | {_kg(row['free_kg'])} | "
            f"{'是' if row['opened'] else '否'} | "
            f"{_kg(row['over_locked_kg']) if row['over_locked_kg'] else '—'} |"
        )
    add("")
    add(report.ledger["rule"])
    add("")

    add("## 7. 产出记录与估计校正")
    add("")
    if report.output_reconciliation:
        add("| 事件 | 工单 | 成品 kg | 已投料 kg | 按估计应出 kg | 差 kg | 实测出汁率 | 估计出汁率 |")
        add("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in report.output_reconciliation:
            add(
                f"| {row['event']} | {row['job']} | {_kg(row['finished_juice_kg'])} | "
                f"{_kg(row['dosed_kg'])} | {_kg(row['dosed_expected_juice_kg'])} | "
                f"{row['output_vs_dosed_estimate_kg']:+.0f} | "
                f"{_fmt(row['implied_yield'])} | {_fmt(row['forecast_yield'])} |"
            )
        add("")
        add("产出只校正已投料部分的出汁率并解释剩料；它不冲减投料，锁料也不会因此自动取消。")
    else:
        add("截止时刻没有成品入库记录。")
    add("")

    add("## 8. 历史试榨复核（开工前证据 vs 后来成品）")
    add("")
    check = report.historical_check[0] if report.historical_check else None
    if check:
        add(
            f"留一法复核 {check['n']} 次试榨：糖度 RMSE {check['brix_rmse']:.2f} °Bx，"
            f"酸度 RMSE {check['acid_rmse']:.4f} 个百分点，出汁率 RMSE {check['yield_rmse']:.3f}。"
        )
        add("")
        rows = check["rows"]
        worst = sorted(
            rows,
            key=lambda r: abs(r["brix_error"]) / 0.1
            + abs(r["acid_error"]) / 0.01
            + abs(r["yield_error"]) / 0.03,
            reverse=True,
        )[:8]
        add("偏差最大的 8 次（完整逐条见 `Report.historical_check`）：")
        add("")
        add("| 试榨 | 种类 | 线 | 开工前证据 | 成品糖度 预测/实际 | 成品酸度 预测/实际 | 出汁率 预测/实际 |")
        add("|---|---|---|---|---|---|---|")
        for row in sorted(worst, key=lambda r: r["pilot"]):
            add(
                f"| {row['pilot']} | {row['kind']} | {row['line']} | {row['evidence_at_start']} | "
                f"{row['predicted_brix_out']:.2f} / {row['actual_brix_out']:.2f} | "
                f"{row['predicted_acid_out']:.3f} / {row['actual_acid_out']:.3f} | "
                f"{row['predicted_yield']:.3f} / {row['actual_yield']:.3f} |"
            )
        add("")
        add(check["rule"])
    add("")

    add("## 9. 配方与价差口径")
    add("")
    for note in report.recipe_notes:
        finding = note["finding"]
        if isinstance(finding, dict):
            add(f"- **{note['topic']}**：")
            add("")
            add("  | 批次 | 种类 | 每 100 kg 出汁 kg | 成品糖度 | 成品酸度 | 证据 |")
            add("  |---|---:|---:|---:|---:|---|")
            for lot_id, row in sorted(finding.items()):
                add(
                    f"  | {lot_id} | {row['kind']} | {row['juice_per_100kg']:.1f} | "
                    f"{row['brix_out']:.2f} | {row['acid_out']:.3f} | "
                    f"{_status_text(row['evidence'])} |"
                )
            add("")
        else:
            add(f"- **{note['topic']}**：{finding}")
        add(f"  - 比较口径：{note['how_to_compare']}")
    add("")

    add("## 10. 模型与覆盖")
    add("")
    for text in report.coverage:
        add(f"- {text}")
    add(f"- 试榨样本：{report.pilot_model['n_pilots']} 次")
    add("")

    if report.change_log:
        add("## 11. 与上一快照相比需要改动的工单")
        add("")
        for line in report.change_log:
            add(f"- {line}")
        add("")

    if report.evidence_requests:
        add("## 12. 需要补充的证据")
        add("")
        for request in report.evidence_requests:
            add(f"- {request}")
        add("")

    add("## 13. 使用约束")
    add("")
    for caveat in report.caveats:
        add(f"- {caveat}")
    add("")
    if report.excluded_events:
        add("证据截止之后才可见、未参与本次计算：")
        for row in report.excluded_commitments:
            add(
                f"- {row['event']} 工单 {row['job']} {row['state']}"
                + (f" 批次 {row['lot']}" if row["lot"] else "")
                + f" {row['qty_kg']:g} kg（记录于 {row['known_at']}）"
            )
        add("")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _status_text(status: str) -> str:
    return {
        "covered": "试榨覆盖",
        "extrapolated": "外推（降级使用）",
        "uncovered": "无试榨（暂估）",
    }.get(status, status)


def _render_plan(add, report: Report, plan) -> None:
    add("| 工单 | 槽/线 | 状态 | 用料 | 原料 kg | 成品汁 kg | 最低 kg | 缺口 kg | 成品糖度 | 成品酸度 |")
    add("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for job in plan.jobs:
        add(
            f"| {job.job} | {job.slot} | {STATUS_LABEL.get(job.status, job.status)} | "
            f"{_mix(dict(job.lots)) or '—'} | {_kg(job.raw_kg)} | {_kg(job.juice_kg)} | "
            f"{_kg(job.minimum_juice_kg)} | {_kg(job.shortfall_kg)} | "
            f"{_num(job.brix_out)} | {_num(job.acid_out)} |"
        )
    add("")
    add("证据链（每张工单的料 → 批次版本 → 袋/复测 → 试榨覆盖）：")
    add("")
    for job in plan.jobs:
        if not job.lots:
            continue
        parts = []
        for lot_id in sorted(job.lots):
            card = next(c for c in report.lots if c.lot == lot_id)
            versions = "、".join(
                f"{bag['prep']} v{bag['version']}" + ("(复测)" if bag["superseded"] else "")
                for bag in card.bags
            )
            parts.append(
                f"{lot_id} rev{card.revision}（{_status_text(card.quality_status)}；{versions}）"
            )
        add(f"- **{job.job}**：" + "；".join(parts))
    add("")
    if plan.leftover:
        add(
            "本方案剩余未用："
            + "、".join(f"{lot} {_kg(qty)} kg" for lot, qty in plan.leftover.items())
        )
        add("")
        add(
            "剩料延续原批证据：已开袋批次的余料若含暂估/外推原料，"
            "仍按原证据等级使用，不当作从未开袋的新货。"
        )
        add("")
    for job in plan.jobs:
        if not job.notes:
            continue
        add(f"- **{job.job}**（{STATUS_LABEL.get(job.status, job.status)}）")
        for note in job.notes:
            add(f"  - {note}")
        for action in job.follow_up:
            add(f"  - 补验/动作：{action}")
        add(f"  - 降档：{job.downgrade}")


def _num(value: float) -> str:
    if value != value:  # NaN
        return "—"
    return f"{value:.2f}"
