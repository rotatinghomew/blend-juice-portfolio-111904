"""End-to-end checks of the command line output."""

import contextlib
import io
import unittest
from pathlib import Path

from blendplan.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_first_snapshot(self):
        code, out, err = run(
            ["--root", str(ROOT), "--as-of", "2026-11-10T09:00:00+08:00"]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("# 混配前分析", out)
        self.assertIn("D5", out)
        self.assertIn("条件接单", out)
        self.assertIn("湖新", out)

    def test_second_snapshot_lists_changes(self):
        code, out, err = run(
            [
                "--root",
                str(ROOT),
                "--as-of",
                "2026-11-11T08:30:00+08:00",
                "--previous",
                "2026-11-10T09:00:00+08:00",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("过磅修正", out)
        self.assertIn("rev1", out)
        self.assertIn("已投料不可回退", out)
        self.assertIn("工单需改", out)

    def test_bad_timestamp(self):
        code, _, err = run(["--root", str(ROOT), "--as-of", "not-a-time"])
        self.assertEqual(code, 2)
        self.assertIn("时间格式错误", err)

    def test_closed_slot_changes_plan(self):
        code, out, _ = run(
            [
                "--root",
                str(ROOT),
                "--as-of",
                "2026-11-11T07:30:00+08:00",
                "--closed-slot",
                "S1",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("已过窗口", out)


if __name__ == "__main__":
    unittest.main()
