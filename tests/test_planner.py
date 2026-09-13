import tempfile
import unittest
from pathlib import Path

from harness.planner import Planner, RepeatGuard


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.plan_path = Path(self._tmp.name) / "PLAN.md"
        self.planner = Planner(self.plan_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_update_writes_file(self):
        result = self.planner.update(summary="do the thing", current="step 1", next_steps=["step 2"])
        self.assertTrue(result["ok"])
        self.assertTrue(self.plan_path.exists())
        content = self.plan_path.read_text(encoding="utf-8")
        self.assertIn("do the thing", content)
        self.assertIn("step 1", content)
        self.assertIn("step 2", content)

    def test_identical_update_is_a_noop(self):
        self.planner.update(summary="s", current="c")
        mtime_before = self.plan_path.stat().st_mtime_ns
        result = self.planner.update(summary="s", current="c")
        self.assertIn("unchanged", result["output"])
        self.assertEqual(self.plan_path.stat().st_mtime_ns, mtime_before)

    def test_loads_existing_plan_state_on_init(self):
        self.planner.update(summary="s", current="c")
        reloaded = Planner(self.plan_path)
        result = reloaded.update(summary="s", current="c")
        self.assertIn("unchanged", result["output"])


class TestRepeatGuard(unittest.TestCase):
    def test_first_call_has_streak_one(self):
        guard = RepeatGuard()
        self.assertEqual(guard.record_and_check("read_file", {"path": "a.py"}), 1)

    def test_repeated_identical_calls_increment_streak(self):
        guard = RepeatGuard()
        guard.record_and_check("read_file", {"path": "a.py"})
        guard.record_and_check("read_file", {"path": "a.py"})
        streak = guard.record_and_check("read_file", {"path": "a.py"})
        self.assertEqual(streak, 3)

    def test_different_args_reset_streak(self):
        guard = RepeatGuard()
        guard.record_and_check("read_file", {"path": "a.py"})
        streak = guard.record_and_check("read_file", {"path": "b.py"})
        self.assertEqual(streak, 1)


if __name__ == "__main__":
    unittest.main()
