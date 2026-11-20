"""Loading, as-of snapshots and evidence grouping.

Design rules enforced here (they are the traps in the source data):

* a reading belongs to a *bag* (``prep``); replicates inside a bag are not
  independent observations of a truck;
* a newer ``version`` replaces the earlier reading of the *same* bag -- two
  versions of one bag are never two samples;
* a lot revision is only visible after ``known_at``; before that the older
  ``net_kg`` applies;
* a commitment only exists after its ``known_at``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Mapping

TS_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class DataError(ValueError):
    """Raised when source records cannot be interpreted consistently."""


def parse_ts(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise DataError(f"bad timestamp {value!r}") from exc


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    if not rows:
        raise DataError(f"{path} is empty")
    return rows


@dataclass(frozen=True)
class Pilot:
    pilot: str
    kind: str
    line: str
    raw_kg: float
    started_at: datetime
    agreed_at: datetime


@dataclass(frozen=True)
class PilotOutput:
    pilot: str
    version: int
    usable_juice_kg: float
    brix: float
    acid_percent: float
    loss_kg: float
    known_at: datetime


@dataclass(frozen=True)
class Reading:
    sample: str
    layer: str
    prep: str
    replicate: int
    version: int
    brix: float
    acid_percent: float
    known_at: datetime


@dataclass(frozen=True)
class RawRevision:
    lot: str
    kind: str
    rev: int
    net_kg: float
    splittable: bool
    expires: date
    known_at: datetime


@dataclass(frozen=True)
class Job:
    job: str
    grade: str
    minimum_juice_kg: float
    min_brix: float
    max_brix: float
    min_acid: float
    max_acid: float
    last_slot: str
    customer_allows_lower: bool


@dataclass(frozen=True)
class Slot:
    slot: str
    line: str
    day: date
    period: str
    max_raw_kg: float
    tank_available: bool
    cleanup_allowed: bool

    @property
    def key(self) -> tuple[date, int]:
        return (self.day, 0 if self.period == "上午" else 1)


@dataclass(frozen=True)
class Commitment:
    event: str
    job: str
    slot: str | None
    lot: str | None
    qty_kg: float
    state: str
    known_at: datetime


@dataclass(frozen=True)
class BagEvidence:
    """One bag (``prep``) at one version, mean over its replicates."""

    sample: str
    prep: str
    layer: str
    version: int
    brix: float
    acid_percent: float
    replicates: int
    known_at: datetime
    superseded: tuple[tuple[int, float, float, datetime], ...] = ()

    @property
    def was_retested(self) -> bool:
        return bool(self.superseded)


@dataclass(frozen=True)
class SampleQuality:
    """A truck / pilot built from bags, never from raw replicate count."""

    sample: str
    bags: tuple[BagEvidence, ...]
    brix: float
    acid_percent: float
    se_brix: float | None
    se_acid: float | None
    between_bag_brix_sd: float | None
    between_bag_acid_sd: float | None
    known_at: datetime
    partial_retest: bool
    version_conflict: bool
    notes: tuple[str, ...] = ()

    @property
    def n_bags(self) -> int:
        return len(self.bags)

    @property
    def scenario_brix_low(self) -> float:
        """Worst whole-truck level supported by the evidence.

        For a truck whose bags disagree after a partial re-test we take the
        lowest bag mean (a *bound*, not an estimate) so that a high reading on
        one layer cannot be presented as the whole truck.
        """

        values = [bag.brix for bag in self.bags]
        for bag in self.bags:
            values.extend(item[1] for item in bag.superseded)
        return min(values)


@dataclass(frozen=True)
class Lot:
    lot: str
    kind: str
    rev: int
    net_kg: float
    splittable: bool
    expires: date
    known_at: datetime
    quality: SampleQuality

    @property
    def free_kg(self) -> float:
        return self.net_kg


@dataclass(frozen=True)
class Dataset:
    root: Path
    pilots: tuple[Pilot, ...]
    pilot_outputs: tuple[PilotOutput, ...]
    readings: tuple[Reading, ...]
    raw_revisions: tuple[RawRevision, ...]
    jobs: tuple[Job, ...]
    slots: tuple[Slot, ...]
    commitments: tuple[Commitment, ...]
    latest_known_at: datetime
    _pilots_by_id: Mapping[str, Pilot] = field(repr=False, default_factory=dict)
    _jobs_by_id: Mapping[str, Job] = field(repr=False, default_factory=dict)
    _slots_by_id: Mapping[str, Slot] = field(repr=False, default_factory=dict)
    _readings_by_sample: Mapping[str, tuple[Reading, ...]] = field(
        repr=False, default_factory=dict
    )
    _revisions_by_lot: Mapping[str, tuple[RawRevision, ...]] = field(
        repr=False, default_factory=dict
    )

    # -- construction -------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> "Dataset":
        root = Path(root)
        pilots = tuple(
            Pilot(
                pilot=r["pilot"],
                kind=r["kind"],
                line=r["line"],
                raw_kg=float(r["raw_kg"]),
                started_at=parse_ts(r["started_at"]),
                agreed_at=parse_ts(r["agreed_at"]),
            )
            for r in _rows(root / "data" / "pilots.csv")
        )
        outputs = tuple(
            PilotOutput(
                pilot=r["pilot"],
                version=int(r["version"]),
                usable_juice_kg=float(r["usable_juice_kg"]),
                brix=float(r["brix"]),
                acid_percent=float(r["acid_percent"]),
                loss_kg=float(r["loss_kg"]),
                known_at=parse_ts(r["known_at"]),
            )
            for r in _rows(root / "data" / "pilot_output.csv")
        )
        readings = tuple(
            Reading(
                sample=r["sample"],
                layer=r["layer"],
                prep=r["prep"],
                replicate=int(r["replicate"]),
                version=int(r["version"]),
                brix=float(r["brix"]),
                acid_percent=float(r["acid_percent"]),
                known_at=parse_ts(r["known_at"]),
            )
            for r in _rows(root / "data" / "readings.csv")
        )
        raw_revisions = tuple(
            RawRevision(
                lot=r["lot"],
                kind=r["kind"],
                rev=int(r["rev"]),
                net_kg=float(r["net_kg"]),
                splittable=r["splittable"] == "1",
                expires=date.fromisoformat(r["expires"]),
                known_at=parse_ts(r["known_at"]),
            )
            for r in _rows(root / "data" / "raw.csv")
        )
        jobs = tuple(
            Job(
                job=r["job"],
                grade=r["grade"],
                minimum_juice_kg=float(r["minimum_juice_kg"]),
                min_brix=float(r["min_brix"]),
                max_brix=float(r["max_brix"]),
                min_acid=float(r["min_acid"]),
                max_acid=float(r["max_acid"]),
                last_slot=r["last_slot"],
                customer_allows_lower=r["customer_allows_lower"] == "1",
            )
            for r in _rows(root / "data" / "jobs.csv")
        )
        slots = tuple(
            Slot(
                slot=r["slot"],
                line=r["line"],
                day=date.fromisoformat(r["day"]),
                period=r["period"],
                max_raw_kg=float(r["max_raw_kg"]),
                tank_available=r["tank_available"] == "1",
                cleanup_allowed=r["cleanup_allowed"] == "1",
            )
            for r in _rows(root / "data" / "slots.csv")
        )
        commitments = tuple(
            Commitment(
                event=r["event"],
                job=r["job"],
                slot=r["slot"] or None,
                lot=r["lot"] or None,
                qty_kg=float(r["qty_kg"]),
                state=r["state"],
                known_at=parse_ts(r["known_at"]),
            )
            for r in _rows(root / "data" / "commitments.csv")
        )

        readings_by_sample: dict[str, list[Reading]] = {}
        for reading in readings:
            readings_by_sample.setdefault(reading.sample, []).append(reading)
        revisions_by_lot: dict[str, list[RawRevision]] = {}
        for revision in raw_revisions:
            revisions_by_lot.setdefault(revision.lot, []).append(revision)

        latest = max(
            [p.agreed_at for p in pilots]
            + [p.started_at for p in pilots]
            + [o.known_at for o in outputs]
            + [r.known_at for r in readings]
            + [r.known_at for r in raw_revisions]
            + [c.known_at for c in commitments]
        )

        return cls(
            root=root,
            pilots=pilots,
            pilot_outputs=outputs,
            readings=readings,
            raw_revisions=raw_revisions,
            jobs=jobs,
            slots=slots,
            commitments=commitments,
            latest_known_at=latest,
            _pilots_by_id={p.pilot: p for p in pilots},
            _jobs_by_id={j.job: j for j in jobs},
            _slots_by_id={s.slot: s for s in slots},
            _readings_by_sample={
                k: tuple(sorted(v, key=lambda r: (r.prep, r.version, r.replicate)))
                for k, v in readings_by_sample.items()
            },
            _revisions_by_lot={
                k: tuple(sorted(v, key=lambda r: r.rev))
                for k, v in revisions_by_lot.items()
            },
        )

    # -- lookups ------------------------------------------------------
    @property
    def pilot_ids(self) -> tuple[str, ...]:
        return tuple(p.pilot for p in self.pilots)

    @property
    def lot_ids(self) -> tuple[str, ...]:
        return tuple(self._revisions_by_lot)

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(j.job for j in self.jobs)

    def pilot(self, pilot_id: str) -> Pilot:
        return self._pilots_by_id[pilot_id]

    def job(self, job_id: str) -> Job:
        return self._jobs_by_id[job_id]

    def slot(self, slot_id: str) -> Slot:
        return self._slots_by_id[slot_id]

    def pilot_output(self, pilot_id: str) -> PilotOutput:
        """Latest output version for a pilot (regardless of time)."""

        outputs = [o for o in self.pilot_outputs if o.pilot == pilot_id]
        if not outputs:
            raise DataError(f"pilot {pilot_id} has no output record")
        return max(outputs, key=lambda o: o.version)

    # -- evidence -----------------------------------------------------
    def bags(self, sample: str, as_of: datetime) -> tuple[BagEvidence, ...]:
        """Latest *known* version of every bag of ``sample``.

        A re-tested bag contributes exactly one row (the newer version); the
        superseded values are attached for audit but never averaged in.
        """

        readings = self._readings_by_sample.get(sample)
        if not readings:
            raise DataError(f"no readings for {sample!r}")
        by_prep: dict[str, list[Reading]] = {}
        for reading in readings:
            if reading.known_at <= as_of:
                by_prep.setdefault(reading.prep, []).append(reading)
        if not by_prep:
            raise DataError(f"no readings for {sample!r} known at {as_of.isoformat()}")

        bags: list[BagEvidence] = []
        for prep in sorted(by_prep):
            rows = by_prep[prep]
            latest_version = max(r.version for r in rows)
            current = [r for r in rows if r.version == latest_version]
            superseded = tuple(
                (
                    version,
                    mean(r.brix for r in rows if r.version == version),
                    mean(r.acid_percent for r in rows if r.version == version),
                    min(r.known_at for r in rows if r.version == version),
                )
                for version in sorted({r.version for r in rows})
                if version < latest_version
            )
            bags.append(
                BagEvidence(
                    sample=sample,
                    prep=prep,
                    layer=current[0].layer,
                    version=latest_version,
                    brix=mean(r.brix for r in current),
                    acid_percent=mean(r.acid_percent for r in current),
                    replicates=len({r.replicate for r in current}),
                    known_at=max(r.known_at for r in current),
                    superseded=superseded,
                )
            )
        return tuple(bags)

    def quality(self, sample: str, as_of: datetime) -> SampleQuality:
        bags = self.bags(sample, as_of)
        brix_values = [b.brix for b in bags]
        acid_values = [b.acid_percent for b in bags]
        notes: list[str] = []
        versions = {b.version for b in bags}
        partial_retest = len(versions) > 1 or any(b.was_retested for b in bags)
        conflict = False
        for bag in bags:
            for version, old_brix, old_acid, _ in bag.superseded:
                if abs(old_brix - bag.brix) > 0.5 or abs(old_acid - bag.acid_percent) > 0.05:
                    conflict = True
                    notes.append(
                        f"{bag.prep} 由 v{version} 改为 v{bag.version}："
                        f"糖度 {old_brix:.2f}→{bag.brix:.2f}，"
                        f"酸度 {old_acid:.3f}→{bag.acid_percent:.3f}"
                    )
        if len(bags) > 1:
            spread = max(brix_values) - min(brix_values)
            notes.append(
                f"{len(bags)} 袋糖度极差 {spread:.2f} °Bx，整车取袋均值，不取最高袋"
            )
        if partial_retest and not conflict:
            notes.append("部分复测：并非每袋都重测")
        if partial_retest:
            notes.append("版本是同袋替换，不是新的整车观测")
        known_at = max(b.known_at for b in bags)
        return SampleQuality(
            sample=sample,
            bags=bags,
            brix=mean(brix_values),
            acid_percent=mean(acid_values),
            se_brix=stdev(brix_values) / len(brix_values) ** 0.5 if len(bags) > 1 else None,
            se_acid=stdev(acid_values) / len(acid_values) ** 0.5 if len(bags) > 1 else None,
            between_bag_brix_sd=stdev(brix_values) if len(bags) > 1 else None,
            between_bag_acid_sd=stdev(acid_values) if len(bags) > 1 else None,
            known_at=known_at,
            partial_retest=partial_retest,
            version_conflict=conflict,
            notes=tuple(notes),
        )

    def lot_state(self, lot: str, as_of: datetime) -> Lot | None:
        revisions = [
            r for r in self._revisions_by_lot.get(lot, ()) if r.known_at <= as_of
        ]
        if not revisions:
            return None
        revision = max(revisions, key=lambda r: r.rev)
        quality = self.quality(lot, as_of)
        return Lot(
            lot=lot,
            kind=revision.kind,
            rev=revision.rev,
            net_kg=revision.net_kg,
            splittable=revision.splittable,
            expires=revision.expires,
            known_at=revision.known_at,
            quality=quality,
        )

    def lots(self, as_of: datetime) -> dict[str, Lot]:
        result: dict[str, Lot] = {}
        for lot in self.lot_ids:
            state = self.lot_state(lot, as_of)
            if state is not None:
                result[lot] = state
        return result

    def visible_commitments(self, as_of: datetime) -> tuple[Commitment, ...]:
        return tuple(c for c in self.commitments if c.known_at <= as_of)


def load_dataset(root: str | Path = ".") -> Dataset:
    return Dataset.load(root)
