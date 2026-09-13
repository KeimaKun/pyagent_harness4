"""Parsing for the agent's prompted JSON-action protocol.

Many small/local OpenAI-API-compatible servers do not reliably support
native `tools=[...]` function calling. Rather than depend on that, the
system prompt instructs the model to reply with exactly one fenced JSON
action block per turn, either:

    {"thought": "...", "tool": "<name>", "args": {...}}
    {"thought": "...", "final": "<answer text for the user>"}

This module extracts and validates that JSON, tolerating minor formatting
slips (extra prose around the block, missing/wrong fence language, etc.).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_ACTION_BLOCK_RE = re.compile(r"```(?:action|json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Action:
    kind: str  # "tool", "final", or "invalid"
    thought: str = ""
    tool: str | None = None
    args: dict[str, Any] | None = None
    final_text: str | None = None
    error: str | None = None
    raw: str = ""


def _try_parse(candidate: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_action(text: str) -> Action:
    raw = text.strip()

    candidates: list[str] = []
    blocks = _ACTION_BLOCK_RE.findall(raw)
    if blocks:
        candidates.append(blocks[-1])  # prefer the last fenced block
    bare = _BARE_OBJECT_RE.findall(raw)
    if bare:
        candidates.append(bare[-1])
    candidates.append(raw)

    obj = None
    for c in candidates:
        obj = _try_parse(c)
        if obj is not None:
            break

    if obj is None:
        return Action(kind="invalid", raw=raw,
                      error="No valid JSON action object found in the response.")

    thought = str(obj.get("thought", ""))

    if "final" in obj:
        return Action(kind="final", thought=thought, final_text=str(obj["final"]), raw=raw)

    if "tool" in obj:
        tool = obj.get("tool")
        args = obj.get("args", {})
        if not isinstance(tool, str) or not tool:
            return Action(kind="invalid", raw=raw, thought=thought,
                          error="'tool' must be a non-empty string.")
        if not isinstance(args, dict):
            return Action(kind="invalid", raw=raw, thought=thought,
                          error="'args' must be a JSON object.")
        return Action(kind="tool", thought=thought, tool=tool, args=args, raw=raw)

    return Action(kind="invalid", raw=raw, thought=thought,
                  error="JSON object must contain either 'tool' or 'final'.")
