"""Filesystem and command sandboxing.

Two different guarantees are made here, and they are NOT equally strong:

1. File tool path containment (`safe_path`) is a hard boundary. Every file
   tool goes through this function, which resolves the path (following
   symlinks / normalizing ".." segments) and refuses anything that does
   not land inside the sandbox root. This cannot be bypassed by the model.

2. Shell command containment (`check_command`) is best-effort static
   filtering only. A general-purpose shell command is Turing-complete and
   a real jail would require OS-level isolation (containers, a restricted
   token / AppContainer, a VM, etc.), which this lightweight CLI harness
   does not attempt. What is actually done: the subprocess `cwd` is always
   pinned to the sandbox root, and command strings containing obvious
   escape patterns (parent-directory traversal, absolute paths outside
   the root, drive switches, UNC paths, env-var expansion, common
   "living off the land" system binaries) are rejected outright. A
   sufficiently adversarial command could still defeat this - treat
   run_command/run_python as "hard to escape by accident", not
   "impossible to escape on purpose".
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path


class SandboxViolation(Exception):
    """Raised whenever an action would touch something outside the sandbox root."""


@dataclass
class CommandCheck:
    ok: bool
    reason: str = ""


# Patterns that, if present in a command string, are treated as an attempted
# escape from the sandbox root. Deliberately conservative (false positives
# over false negatives).
_ESCAPE_PATTERNS = [
    re.compile(r"\.\.[\\/]"),                       # ../ or ..\
    re.compile(r"^\s*[a-zA-Z]:\\?\s*$"),            # bare drive switch, e.g. "D:"
    re.compile(r"\\\\[^\\]"),                       # UNC path \\server\share
    re.compile(r"%[A-Za-z_][A-Za-z0-9_]*%"),        # Windows env var expansion %VAR%
    re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"),  # $VAR / ${VAR} expansion
    re.compile(r"~[\\/]?"),                         # home-dir shorthand
]

# Commands that are never allowed regardless of arguments, because they act
# on system/global state rather than the project directory, or are common
# "living off the land" binaries used to fetch/execute things outside a
# sandbox (certutil/bitsadmin/mshta/regsvr32, etc.).
_BLOCKED_COMMAND_NAMES = {
    "shutdown", "reboot", "restart-computer", "diskpart", "format",
    "reg", "regedit", "bcdedit", "vssadmin", "net", "netsh",
    "takeown", "icacls", "sc", "wmic", "cipher", "fsutil",
    "certutil", "bitsadmin", "mshta", "regsvr32", "wevtutil",
    "schtasks", "taskkill",
}


class Sandbox:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise SandboxViolation(f"Sandbox root is not a directory: {self.root}")

    def safe_path(self, user_path: str | os.PathLike, must_exist: bool = False) -> Path:
        """Resolve user_path against the sandbox root and verify containment.

        Accepts both relative paths (resolved against root) and absolute
        paths (must themselves already be inside root). Raises
        SandboxViolation on any attempt to leave the root.
        """
        raw = str(user_path).strip()
        if not raw or raw in (".", "./"):
            candidate = self.root
        else:
            p = Path(raw)
            candidate = p if p.is_absolute() else (self.root / p)

        # Resolve without requiring existence, so new files can be created.
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as e:
            raise SandboxViolation(f"Could not resolve path {raw!r}: {e}") from e

        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise SandboxViolation(
                f"Path {raw!r} resolves to {resolved}, which is outside the "
                f"sandbox root {self.root}"
            )

        if must_exist and not resolved.exists():
            raise SandboxViolation(f"Path does not exist: {resolved}")

        return resolved

    def rel(self, path: Path) -> str:
        """Render an absolute in-sandbox path relative to the root, for display."""
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def check_command(self, command: str) -> CommandCheck:
        """Best-effort static check that `command` does not try to escape root."""
        if not command or not command.strip():
            return CommandCheck(False, "Empty command.")

        for pattern in _ESCAPE_PATTERNS:
            if pattern.search(command):
                return CommandCheck(
                    False,
                    f"Command rejected: matched disallowed pattern {pattern.pattern!r} "
                    "(looks like it may reference a path outside the project, "
                    "an environment-variable expansion, or a drive switch).",
                )

        try:
            tokens = shlex.split(command, posix=False)
        except ValueError as e:
            return CommandCheck(False, f"Could not parse command: {e}")

        for tok in tokens:
            bare = tok.strip("\"'").lower()
            name = os.path.basename(bare)
            if name.endswith(".exe"):
                name = name[: -len(".exe")]
            if name in _BLOCKED_COMMAND_NAMES:
                return CommandCheck(False, f"Command '{name}' is not permitted.")
            # Any token that looks like an absolute path must resolve inside root.
            if re.match(r"^[a-zA-Z]:[\\/]", bare) or bare.startswith("\\\\"):
                try:
                    self.safe_path(tok.strip("\"'"))
                except SandboxViolation as e:
                    return CommandCheck(False, str(e))

        return CommandCheck(True)
