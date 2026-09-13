"""Plan state management + anti-repetition guard.

`Planner` atomically overwrites a single PLAN.md each time the agent calls
`update_plan`, keeping only the latest state rather than an append-only
log that would bloat the context window - the same "overwrite state"
approach Antigravity's own task/plan panel uses. Writing the identical
plan twice in a row is a no-op (and reported as such), which discourages
the model from padding turns with no-op plan updates.

`RepeatGuard` tracks recent (tool, args) tool calls and flags when the
agent is stuck calling the exact same thing over and over with no new
information - a common failure mode for smaller/local models - so the
agent loop can nudge it, and eventually abort, instead of looping forever.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any


class Planner:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path
        self._last_content: str | None = None
        if plan_path.exists():
            try:
                self._last_content = plan_path.read_text(encoding="utf-8")
            except OSError:
                pass

    @staticmethod
    def render(summary: str, current: str, completed: list[str], next_steps: list[str]) -> str:
        def _bullets(items: list[str], box: str) -> str:
            return "\n".join(f"- [{box}] {item}" for item in items) if items else "- (none)"

        return (
            "# Plan\n\n"
            f"## Summary\n{summary or '(none)'}\n\n"
            "## Current step\n" + (f"- [ ] {current}" if current else "- (none)") + "\n\n"
            "## Completed\n" + _bullets(completed, "x") + "\n\n"
            "## Up next\n" + _bullets(next_steps, " ") + "\n"
        )

    def update(
        self,
        summary: str = "",
        current: str = "",
        completed: list[str] | None = None,
        next_steps: list[str] | None = None,
    ) -> dict[str, Any]:
        content = self.render(summary, current, completed or [], next_steps or [])
        if content == self._last_content:
            return {"ok": True, "output": "Plan unchanged (identical to current state); no write performed."}

        tmp = self.plan_path.with_suffix(self.plan_path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, self.plan_path)
        self._last_content = content
        return {"ok": True, "output": f"Plan updated ({self.plan_path.name})."}


class RepeatGuard:
    """Detects the agent issuing the same (tool, args) call repeatedly in a row."""

    def __init__(self, warn_limit: int = 3, window: int = 8):
        self.warn_limit = warn_limit
        self.recent: deque[str] = deque(maxlen=window)

    @staticmethod
    def _key(tool: str, args: dict) -> str:
        try:
            return tool + "|" + json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            return tool + "|" + str(args)

    def record_and_check(self, tool: str, args: dict) -> int:
        """Records this call and returns how many times it has now repeated
        consecutively (1 = first time / no repeat)."""
        key = self._key(tool, args)
        self.recent.append(key)
        streak = 0
        for k in reversed(self.recent):
            if k == key:
                streak += 1
            else:
                break
        return streak
