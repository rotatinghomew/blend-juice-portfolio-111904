"""Pilot evidence -> lot forecasts.

Every equation is fitted on *truck-level* pilot rows built from bag means.
Replicates inside one bag are never treated as independent truck samples,
and a re-tested bag contributes its newest version only.

Relations kept together
-----------------------
1. ``acid_in = a0 + a1 * brix_in`` -- the acid/brix line of incoming fruit.
2. ``yield = y0 + y1 * brix_in + kind effects`` -- usable juice / raw.
3. ``brix_out = b0 + b1 * brix_in + b2 * acid_gap`` with
   ``acid_gap = acid_in - line(brix_in)``.
4. ``acid_out = c0 + c1 * brix_out``.

A lot's low/high scenarios are evaluated by *rebuilding* all three outputs
from the same bag reading: a low-brix bag gives a low yield, low finished
brix and (because ``c1 < 0``) high finished acid together.  The best reading
and the best yield of different bags are never combined.

``loss_kg`` is a separate source quantity and is never added to
``usable_juice_kg``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from .data import Dataset, Lot, Pilot, PilotOutput, SampleQuality

MIN_PILOTS = 24


class NotFitted(RuntimeError):
    pass


@dataclass(frozen=True)
class LinearFit:
    names: tuple[str, ...]
    beta: np.ndarray
    xtx_inv: np.ndarray
    resid_sd: float
    dof: int
    n: int

    def row(self, values: Mapping[str, float]) -> np.ndarray:
        return np.array([1.0] + [float(values[name]) for name in self.names], dtype=float)

    def value(self, values: Mapping[str, float]) -> float:
        return float(self.row(values) @ self.beta)

    def prediction_sd(self, values: Mapping[str, float]) -> float:
        x = self.row(values)
        var = self.resid_sd**2 * (1.0 + float(x @ self.xtx_inv @ x))
        return sqrt(max(var, 0.0))

    def t_stat(self, index: int) -> float:
        var = self.resid_sd**2 * self.xtx_inv[index, index]
        return float(self.beta[index] / sqrt(var)) if var > 0 else float("nan")


def _fit(
    names: tuple[str, ...], rows: Sequence[Mapping[str, float]], target: Sequence[float]
) -> LinearFit:
    design = np.array([[1.0] + [row[name] for name in names] for row in rows], dtype=float)
    y = np.array(target, dtype=float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    dof = max(len(y) - design.shape[1], 1)
    resid_sd = sqrt(float(residuals @ residuals) / dof)
    return LinearFit(
        names=names,
        beta=beta,
        xtx_inv=np.linalg.inv(design.T @ design),
        resid_sd=resid_sd,
        dof=dof,
        n=len(y),
    )


@dataclass(frozen=True)
class LineCoverage:
    line: str
    pilots: int
    brix_min: float
    brix_max: float
    acid_gap_min: float
    acid_gap_max: float


@dataclass(frozen=True)
class Coverage:
    kind: str
    pilots: int
    brix_min: float
    brix_max: float
    acid_min: float
    acid_max: float
    acid_gap_min: float
    acid_gap_max: float
    raw_min: float
    raw_max: float
    by_line: Mapping[str, LineCoverage]

    def contains(self, brix: float, acid_gap: float) -> bool:
        return (
            self.brix_min <= brix <= self.brix_max
            and self.acid_gap_min <= acid_gap <= self.acid_gap_max
        )

    def gap_excess(self, acid_gap: float) -> float:
        if acid_gap > self.acid_gap_max:
            return acid_gap - self.acid_gap_max
        if acid_gap < self.acid_gap_min:
            return self.acid_gap_min - acid_gap
        return 0.0

    def describes_line(self, line: str) -> bool:
        item = self.by_line.get(line)
        return item is not None and item.pilots >= 3

    def described(self) -> str:
        lines = "、".join(
            f"{line} {item.pilots} 次（{item.brix_min:.1f}–{item.brix_max:.1f} °Bx）"
            for line, item in sorted(self.by_line.items())
        )
        return (
            f"{self.pilots} 次试榨，来料糖度 {self.brix_min:.1f}–{self.brix_max:.1f} °Bx，"
            f"酸度 {self.acid_min:.3f}–{self.acid_max:.3f}%，"
            f"酸-糖线偏差 {self.acid_gap_min:+.3f}–{self.acid_gap_max:+.3f}；"
            f"产线：{lines}"
        )


@dataclass(frozen=True)
class Point:
    """One consistent (yield, brix_out, acid_out) triple."""

    yield_: float
    brix_out: float
    acid_out: float


@dataclass(frozen=True)
class LotForecast:
    lot: str
    kind: str
    brix: float
    acid: float
    acid_gap: float
    n_bags: int
    se_brix: float | None
    bag_brix_values: tuple[float, ...]
    bag_acid_values: tuple[float, ...]
    status: str  # covered | extrapolated | uncovered
    coverage_note: str
    point: Point
    point_sd: Point
    low: Point
    high: Point
    scenario_source: str
    notes: tuple[str, ...]

    @property
    def firm_eligible(self) -> bool:
        return self.status == "covered"


@dataclass(frozen=True)
class ReconciliationRow:
    pilot: str
    kind: str
    line: str
    started_at: datetime
    agreed_at: datetime
    evidence_at_start: str
    brix_in: float
    acid_in: float
    predicted_brix_out: float
    actual_brix_out: float
    predicted_acid_out: float
    actual_acid_out: float
    predicted_yield: float
    actual_yield: float
    output_known_at: datetime
    brix_error: float
    acid_error: float
    yield_error: float


class PilotModels:
    """Pilot relations frozen at one ``as_of`` evidence cutoff."""

    def __init__(self, dataset: Dataset, as_of: datetime) -> None:
        self.dataset = dataset
        self.as_of = as_of
        self._samples: list[tuple[Pilot, PilotOutput, SampleQuality]] = []
        self._rows: list[dict[str, float]] = []
        self._build_rows()
        if len(self._rows) < MIN_PILOTS:
            raise NotFitted(
                f"only {len(self._rows)} pilots had output known at {as_of.isoformat()}"
            )
        self.acid_line = _fit(("brix",), self._rows, [r["acid"] for r in self._rows])
        for row in self._rows:
            row["acid_gap"] = row["acid"] - self.acid_in_line(row["brix"])
        self.yield_fit = _fit(
            ("brix", "is_pinggang", "is_shuitian"),
            self._rows,
            [r["yield"] for r in self._rows],
        )
        self.brix_fit = _fit(
            ("brix", "acid_gap"), self._rows, [r["brix_out"] for r in self._rows]
        )
        self.acid_fit = _fit(
            ("brix_out",), self._rows, [r["acid_out"] for r in self._rows]
        )
        self._coverage = self._coverage_table()

    # -- evidence ------------------------------------------------------
    def _build_rows(self) -> None:
        for pilot in self.dataset.pilots:
            outputs = [
                o
                for o in self.dataset.pilot_outputs
                if o.pilot == pilot.pilot and o.known_at <= self.as_of
            ]
            if not outputs:
                continue
            output = max(outputs, key=lambda o: o.version)
            try:
                quality = self.dataset.quality(pilot.pilot, pilot.started_at)
            except Exception:  # pragma: no cover - incomplete evidence
                continue
            if quality.known_at > pilot.started_at:
                # a reading known only after start cannot explain the start
                continue
            self._samples.append((pilot, output, quality))
            self._rows.append(
                {
                    "brix": quality.brix,
                    "acid": quality.acid_percent,
                    "acid_gap": 0.0,
                    "yield": output.usable_juice_kg / pilot.raw_kg,
                    "brix_out": output.brix,
                    "acid_out": output.acid_percent,
                    "is_pinggang": 1.0 if pilot.kind == "平岗" else 0.0,
                    "is_shuitian": 1.0 if pilot.kind == "水田" else 0.0,
                    "raw": pilot.raw_kg,
                    "loss_fraction": output.loss_kg / pilot.raw_kg,
                }
            )

    def _coverage_table(self) -> dict[str, Coverage]:
        table: dict[str, Coverage] = {}
        for kind in sorted({p.kind for p in self.dataset.pilots}):
            pairs = [
                (row, pilot)
                for row, (pilot, _, _) in zip(self._rows, self._samples)
                if pilot.kind == kind
            ]
            if not pairs:
                continue
            rows = [row for row, _ in pairs]
            by_line: dict[str, list[dict[str, float]]] = {}
            for row, pilot in pairs:
                by_line.setdefault(pilot.line, []).append(row)
            table[kind] = Coverage(
                kind=kind,
                pilots=len(rows),
                brix_min=min(r["brix"] for r in rows),
                brix_max=max(r["brix"] for r in rows),
                acid_min=min(r["acid"] for r in rows),
                acid_max=max(r["acid"] for r in rows),
                acid_gap_min=min(r["acid_gap"] for r in rows),
                acid_gap_max=max(r["acid_gap"] for r in rows),
                raw_min=min(r["raw"] for r in rows),
                raw_max=max(r["raw"] for r in rows),
                by_line={
                    line: LineCoverage(
                        line=line,
                        pilots=len(values),
                        brix_min=min(v["brix"] for v in values),
                        brix_max=max(v["brix"] for v in values),
                        acid_gap_min=min(v["acid_gap"] for v in values),
                        acid_gap_max=max(v["acid_gap"] for v in values),
                    )
                    for line, values in sorted(by_line.items())
                },
            )
        return table

    # -- queries -------------------------------------------------------
    def coverage(self, kind: str) -> Coverage | None:
        return self._coverage.get(kind)

    def kinds(self) -> tuple[str, ...]:
        return tuple(self._coverage)

    def acid_in_line(self, brix: float) -> float:
        return self.acid_line.value({"brix": brix})

    def acid_gap(self, brix: float, acid: float) -> float:
        return acid - self.acid_in_line(brix)

    def _kind(self, kind: str) -> dict[str, float]:
        return {
            "is_pinggang": 1.0 if kind == "平岗" else 0.0,
            "is_shuitian": 1.0 if kind == "水田" else 0.0,
        }

    def yield_at(self, kind: str, brix: float) -> tuple[float, float]:
        row = {"brix": brix, **self._kind(kind)}
        return self.yield_fit.value(row), self.yield_fit.prediction_sd(row)

    def brix_out_at(self, brix: float, gap: float) -> tuple[float, float]:
        row = {"brix": brix, "acid_gap": gap}
        return self.brix_fit.value(row), self.brix_fit.prediction_sd(row)

    def acid_out_at(self, brix_out: float) -> tuple[float, float]:
        row = {"brix_out": brix_out}
        return self.acid_fit.value(row), self.acid_fit.prediction_sd(row)

    def acid_out(self, brix_out: float) -> float:
        return self.acid_fit.value({"brix_out": brix_out})

    def point_at(self, kind: str, brix: float, acid: float) -> tuple[Point, Point]:
        gap = self.acid_gap(brix, acid)
        y, y_sd = self.yield_at(kind, brix)
        b, b_sd = self.brix_out_at(brix, gap)
        a, a_sd = self.acid_out_at(b)
        return Point(y, b, a), Point(y_sd, b_sd, a_sd)

    # -- lot forecast --------------------------------------------------
    def forecast_lot(self, lot: Lot) -> LotForecast:
        quality = lot.quality
        brix = quality.brix
        acid = quality.acid_percent
        gap = self.acid_gap(brix, acid)
        coverage = self.coverage(lot.kind)
        notes: list[str] = list(quality.notes)
        if quality.se_brix is not None:
            notes.append(f"袋间糖度标准误 {quality.se_brix:.2f} °Bx（{quality.n_bags} 袋）")

        if coverage is not None and coverage.contains(brix, gap):
            status = "covered"
            coverage_note = f"糖度与酸-糖线均落在 {lot.kind} 试榨范围内"
        elif coverage is not None:
            status = "extrapolated"
            if not (coverage.brix_min <= brix <= coverage.brix_max):
                coverage_note = (
                    f"来料糖度 {brix:.2f} 超出 {lot.kind} 试榨 "
                    f"{coverage.brix_min:.1f}–{coverage.brix_max:.1f} °Bx"
                )
            else:
                coverage_note = (
                    f"酸-糖线偏差 {gap:+.3f} 超出 {lot.kind} 试榨范围 "
                    f"[{coverage.acid_gap_min:+.3f}, {coverage.acid_gap_max:+.3f}]，"
                    "成品酸度只能外推"
                )
            notes.append("超出可外推范围，需要补试榨/补样后才能承诺")
        else:
            status = "uncovered"
            coverage_note = f"{lot.kind} 没有试榨记录，全部为外推"
            notes.append("新来料没有任何本档试榨证据，不能据此作承诺")

        point, point_sd = self.point_at(lot.kind, brix, acid)
        # A lot's own bags set the whole-lot low/high scenario: the same bag
        # supplies brix and acid, never the best of each.
        bag_brix = [bag.brix for bag in quality.bags]
        bag_acid = [bag.acid_percent for bag in quality.bags]
        low_index = min(range(len(bag_brix)), key=bag_brix.__getitem__)
        high_index = max(range(len(bag_brix)), key=bag_brix.__getitem__)
        low_point, _ = self.point_at(lot.kind, bag_brix[low_index], bag_acid[low_index])
        high_point, _ = self.point_at(lot.kind, bag_brix[high_index], bag_acid[high_index])
        scenario_source = (
            f"最低袋 {quality.bags[low_index].prep} "
            f"({bag_brix[low_index]:.2f} °Bx/{bag_acid[low_index]:.3f}%)，"
            f"最高袋 {quality.bags[high_index].prep} "
            f"({bag_brix[high_index]:.2f} °Bx/{bag_acid[high_index]:.3f}%)"
        )
        if quality.partial_retest:
            notes.append("部分复测：以每袋最新版本为准，未把复测当第二份整车观测")
        if status == "uncovered":
            # No kind prior: widen the scenario to the range of the three
            # known kinds at this brix, and keep the same-bag coupling.
            yields = [self.yield_at(kind, brix)[0] for kind in self.kinds()]
            spread = max(yields) - min(yields)
            low_point = Point(
                min(low_point.yield_, min(yields)) - spread / 2,
                low_point.brix_out,
                low_point.acid_out,
            )
            high_point = Point(
                max(high_point.yield_, max(yields)) + spread / 2,
                high_point.brix_out,
                high_point.acid_out,
            )
            point_sd = Point(
                sqrt(point_sd.yield_**2 + (spread / 2) ** 2),
                point_sd.brix_out,
                point_sd.acid_out,
            )

        # widen each scenario by the model's own prediction error, always
        # outwards (lower yield / lower brix / higher acid on the low side)
        low = Point(
            max(low_point.yield_ - point_sd.yield_, 0.0),
            low_point.brix_out - point_sd.brix_out,
            low_point.acid_out + point_sd.acid_out,
        )
        high = Point(
            high_point.yield_ + point_sd.yield_,
            high_point.brix_out + point_sd.brix_out,
            max(high_point.acid_out - point_sd.acid_out, 0.0),
        )
        return LotForecast(
            lot=lot.lot,
            kind=lot.kind,
            brix=brix,
            acid=acid,
            acid_gap=gap,
            n_bags=quality.n_bags,
            se_brix=quality.se_brix,
            bag_brix_values=tuple(bag_brix),
            bag_acid_values=tuple(bag_acid),
            status=status,
            coverage_note=coverage_note,
            point=point,
            point_sd=point_sd,
            low=low,
            high=high,
            scenario_source=scenario_source,
            notes=tuple(dict.fromkeys(notes)),
        )

    def forecast_lots(self, lots: Mapping[str, Lot]) -> dict[str, LotForecast]:
        return {lot_id: self.forecast_lot(lot) for lot_id, lot in lots.items()}

    # -- diagnostics ---------------------------------------------------
    def line_effect(self) -> dict[str, object]:
        """Line effect after controlling for kind and incoming brix.

        The lines are not randomised across fruit kinds (水田 is almost all on
        榨一), so an uncontrolled difference is composition, not equipment.
        """

        rows = []
        target = []
        for row, (pilot, _, _) in zip(self._rows, self._samples):
            rows.append(
                {
                    "brix": row["brix"],
                    "is_pinggang": row["is_pinggang"],
                    "is_shuitian": row["is_shuitian"],
                    "is_line2": 1.0 if pilot.line == "榨二" else 0.0,
                }
            )
            target.append(row["yield"])
        fit = _fit(("brix", "is_pinggang", "is_shuitian", "is_line2"), rows, target)
        counts: dict[tuple[str, str], int] = {}
        for pilot, _, _ in self._samples:
            counts[(pilot.kind, pilot.line)] = counts.get((pilot.kind, pilot.line), 0) + 1
        return {
            "line2_beta": float(fit.beta[4]),
            "line2_t": fit.t_stat(4),
            "resid_sd": fit.resid_sd,
            "counts": counts,
        }

    def loss_summary(self) -> dict[str, tuple[float, float]]:
        summary: dict[str, list[float]] = {}
        for row, (pilot, _, _) in zip(self._rows, self._samples):
            summary.setdefault(pilot.kind, []).append(row["loss_fraction"])
        return {
            kind: (mean(values), float(np.std(values, ddof=1)))
            for kind, values in summary.items()
        }

    def reconcile(self) -> list[ReconciliationRow]:
        """Check pre-start evidence against the later output, leave-one-out.

        Each pilot is predicted by models fitted *without* that pilot.  The
        input side is frozen at ``started_at``; the output is used only for
        the comparison, never to backfill the pre-start evidence.
        """

        rows: list[ReconciliationRow] = []
        for pilot, output, quality in self._samples:
            others = [r for r, (p, _, _) in zip(self._rows, self._samples) if p.pilot != pilot.pilot]
            if len(others) < MIN_PILOTS:
                continue
            acid_fit = _fit(("brix",), others, [r["acid"] for r in others])
            yield_fit = _fit(
                ("brix", "is_pinggang", "is_shuitian"),
                others,
                [r["yield"] for r in others],
            )
            gap = quality.acid_percent - acid_fit.value({"brix": quality.brix})
            brix_fit = _fit(
                ("brix", "acid_gap"),
                [
                    {**r, "acid_gap": r["acid"] - acid_fit.value({"brix": r["brix"]})}
                    for r in others
                ],
                [r["brix_out"] for r in others],
            )
            acid_out_fit = _fit(("brix_out",), others, [r["acid_out"] for r in others])
            yield_hat = yield_fit.value({"brix": quality.brix, **self._kind(pilot.kind)})
            brix_hat = brix_fit.value({"brix": quality.brix, "acid_gap": gap})
            acid_hat = acid_out_fit.value({"brix_out": brix_hat})
            actual_yield = output.usable_juice_kg / pilot.raw_kg
            rows.append(
                ReconciliationRow(
                    pilot=pilot.pilot,
                    kind=pilot.kind,
                    line=pilot.line,
                    started_at=pilot.started_at,
                    agreed_at=pilot.agreed_at,
                    evidence_at_start=(
                        f"{quality.n_bags} 袋，最后读数 {quality.known_at.isoformat()}"
                    ),
                    brix_in=quality.brix,
                    acid_in=quality.acid_percent,
                    predicted_brix_out=brix_hat,
                    actual_brix_out=output.brix,
                    predicted_acid_out=acid_hat,
                    actual_acid_out=output.acid_percent,
                    predicted_yield=yield_hat,
                    actual_yield=actual_yield,
                    output_known_at=output.known_at,
                    brix_error=brix_hat - output.brix,
                    acid_error=acid_hat - output.acid_percent,
                    yield_error=yield_hat - actual_yield,
                )
            )
        return rows


def fit_models(dataset: Dataset, as_of: datetime) -> PilotModels:
    return PilotModels(dataset, as_of)


def reconcile_pilots(
    dataset: Dataset, as_of: datetime | None = None
) -> list[ReconciliationRow]:
    return PilotModels(dataset, as_of or dataset.latest_known_at).reconcile()
