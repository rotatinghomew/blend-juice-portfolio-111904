"""Command line entry point.

    python3 -m blendplan --as-of 2026-11-10T09:00:00+08:00
    python3 -m blendplan --as-of 2026-11-11T08:30:00+08:00 \\
        --previous 2026-11-10T09:00:00+08:00

Reads the CSVs under ``data/`` only; prints Markdown to stdout.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from .data import DataError, Dataset
from .estimates import NotFitted
from .render import render_markdown
from .report import build_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="混配前分析（只读 data/）")
    parser.add_argument(
        "--root", default=".", help="仓库根目录（默认当前目录）"
    )
    parser.add_argument(
        "--as-of", required=True, help="证据截止时间，ISO 8601（含时区）"
    )
    parser.add_argument(
        "--previous", default=None, help="上一快照时间，用于列出需改工单"
    )
    parser.add_argument(
        "--closed-slot",
        action="append",
        default=[],
        help="本次已关闭、不能再排产的槽（可重复）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        as_of = datetime.fromisoformat(args.as_of)
        previous = datetime.fromisoformat(args.previous) if args.previous else None
    except ValueError as exc:
        print(f"时间格式错误：{exc}", file=sys.stderr)
        return 2
    try:
        dataset = Dataset.load(args.root)
        report = build_report(
            dataset,
            as_of,
            previous_as_of=previous,
            closed_slots=args.closed_slot,
        )
    except (DataError, NotFitted, RuntimeError) as exc:
        print(f"无法生成分析：{exc}", file=sys.stderr)
        return 1
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
