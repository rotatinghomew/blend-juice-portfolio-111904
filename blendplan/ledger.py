"""Two separate ledgers, never netted against each other.

* **Usage ledger** -- ``锁料`` (reservation), ``待批`` (pending work order) and
  ``已投料`` (physically dosed).  Only ``已投料`` is irreversible and only it
  reduces the free quantity of a lot.  A reservation is a *plan*, not
  consumption.
* **Output ledger** -- ``成品入库`` (finished juice).  Output corrects the
  estimate and explains leftovers; it never substitutes for a usage entry.

A partially used lot is not "new material": its remaining kilograms keep the
truck's evidence, are marked as opened, and are never given a fresh-sample
status.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .data import Commitment, Dataset


USAGE_STATES = ("锁料", "待批", "已投料")
IRREVERSIBLE = "已投料"


@dataclass(frozen=True)
class LotUsage:
    lot: str
    net_kg: float
    dosed_kg: float
    reserved_kg: float   # remaining reservation after dosed consumption
    pending_kg: float
    reserved_gross_kg: float = 0.0
    pending_gross_kg: float = 0.0

    @property
    def over_locked_kg(self) -> float:
        """Reservations/pending that no longer fit the lot's weighed weight."""

        return max(self.reserved_kg + self.pending_kg - self.net_kg, 0.0)

    @property
    def free_kg(self) -> float:
        return max(self.net_kg - self.dosed_kg, 0.0)

    @property
    def uncommitted_kg(self) -> float:
        """Free material that is not even reserved."""

        return max(self.net_kg - self.dosed_kg - self.reserved_kg - self.pending_kg, 0.0)

    @property
    def opened(self) -> bool:
        return self.dosed_kg > 0.0


@dataclass(frozen=True)
class JobUsage:
    job: str
    dosed_by_lot: Mapping[str, float]
    reserved_by_lot: Mapping[str, float]
    pending_by_lot: Mapping[str, float]
    finished_juice_kg: float
    events: tuple[Commitment, ...]


@dataclass(frozen=True)
class Ledger:
    as_of: datetime
    lots: Mapping[str, LotUsage]
    jobs: Mapping[str, JobUsage]
    raw_events: tuple[Commitment, ...]

    def job(self, job: str) -> JobUsage:
        return self.jobs[job]

    def lot(self, lot: str) -> LotUsage:
        return self.lots[lot]

    def dosed_for(self, job: str, lot: str) -> float:
        return self.jobs[job].dosed_by_lot.get(lot, 0.0)

    def free_lot_kg(self, lot: str) -> float:
        usage = self.lots.get(lot)
        return usage.free_kg if usage else 0.0


def build_ledger(dataset: Dataset, as_of: datetime) -> Ledger:
    events = tuple(
        c for c in dataset.commitments if c.known_at <= as_of and c.state in USAGE_STATES + ("成品入库",)
    )
    lots: dict[str, dict[str, float]] = {
        lot: {"net": 0.0, "dosed": 0.0, "reserved": 0.0, "pending": 0.0}
        for lot in dataset.lot_ids
    }
    # net kg at this snapshot (latest revision known now), even for a lot with
    # no commitment event yet.
    for lot_id in dataset.lot_ids:
        state = dataset.lot_state(lot_id, as_of)
        if state is not None:
            lots[lot_id]["net"] = state.net_kg

    jobs: dict[str, dict[str, object]] = {}
    for job_id in dataset.job_ids:
        jobs[job_id] = {
            "dosed": {},
            "reserved": {},
            "pending": {},
            "finished": 0.0,
            "events": [],
        }

    for event in events:
        job = jobs.setdefault(
            event.job,
            {"dosed": {}, "reserved": {}, "pending": {}, "finished": 0.0, "events": []},
        )
        job["events"].append(event)
        if event.state == "成品入库":
            job["finished"] += event.qty_kg
            continue
        if not event.lot:
            continue
        bucket = {"锁料": "reserved", "待批": "pending", "已投料": "dosed"}[event.state]
        lot_book = lots.setdefault(
            event.lot, {"net": 0.0, "dosed": 0.0, "reserved": 0.0, "pending": 0.0}
        )
        lot_book[bucket] += event.qty_kg
        job[bucket][event.lot] = job[bucket].get(event.lot, 0.0) + event.qty_kg

    lot_usage = {}
    for lot_id, book in lots.items():
        # ``已投料`` consumes the reservation it was drawn against, so it is
        # not counted twice.  Without a same-job reservation the dosed mass
        # simply reduces free stock.
        remaining_lock = max(book["reserved"] - book["dosed"], 0.0)
        remaining_pending = max(book["pending"] - book["dosed"], 0.0)
        lot_usage[lot_id] = LotUsage(
            lot=lot_id,
            net_kg=book["net"],
            dosed_kg=book["dosed"],
            reserved_kg=remaining_lock,
            pending_kg=remaining_pending,
            reserved_gross_kg=book["reserved"],
            pending_gross_kg=book["pending"],
        )
    job_usage = {
        job_id: JobUsage(
            job=job_id,
            dosed_by_lot=dict(book["dosed"]),
            reserved_by_lot=dict(book["reserved"]),
            pending_by_lot=dict(book["pending"]),
            finished_juice_kg=float(book["finished"]),
            events=tuple(book["events"]),
        )
        for job_id, book in jobs.items()
    }
    return Ledger(as_of=as_of, lots=lot_usage, jobs=job_usage, raw_events=events)
