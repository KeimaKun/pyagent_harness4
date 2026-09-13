import tempfile
import unittest
from pathlib import Path

from harness.sandbox import Sandbox, SandboxViolation


class TestSandboxPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "sub").mkdir()
        (self.root / "sub" / "file.txt").write_text("hi", encoding="utf-8")
        self.sandbox = Sandbox(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_relative_path_inside_root_is_allowed(self):
        p = self.sandbox.safe_path("sub/file.txt", must_exist=True)
        self.assertTrue(str(p).startswith(str(self.root)))

    def test_parent_traversal_is_rejected(self):
        with self.assertRaises(SandboxViolation):
            self.sandbox.safe_path("../outside.txt")

    def test_absolute_path_outside_root_is_rejected(self):
        with self.assertRaises(SandboxViolation):
            self.sandbox.safe_path(str(Path(tempfile.gettempdir()) / "definitely-outside.txt"))

    def test_new_file_path_that_does_not_exist_yet_is_allowed(self):
        p = self.sandbox.safe_path("new_file.txt")
        self.assertTrue(str(p).startswith(str(self.root)))

    def test_must_exist_rejects_missing_path(self):
        with self.assertRaises(SandboxViolation):
            self.sandbox.safe_path("nope.txt", must_exist=True)


class TestCommandChecks(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.sandbox = Sandbox(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_plain_command_is_allowed(self):
        self.assertTrue(self.sandbox.check_command("python --version").ok)

    def test_parent_traversal_in_command_is_rejected(self):
        self.assertFalse(self.sandbox.check_command("type ..\\secrets.txt").ok)

    def test_env_var_expansion_is_rejected(self):
        self.assertFalse(self.sandbox.check_command("echo %APPDATA%").ok)
        self.assertFalse(self.sandbox.check_command("echo $HOME").ok)

    def test_blocked_system_command_is_rejected(self):
        self.assertFalse(self.sandbox.check_command("shutdown /s /t 0").ok)
        self.assertFalse(self.sandbox.check_command("reg query HKLM").ok)

    def test_empty_command_is_rejected(self):
        self.assertFalse(self.sandbox.check_command("   ").ok)


if __name__ == "__main__":
    unittest.main()
