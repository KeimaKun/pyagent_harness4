"""Console rendering, stdlib-only (no `rich` or other third-party UI
library), styled loosely after Antigravity's live agent trace: a running
step-by-step log with thoughts, tool calls, diffs, and a confirmation gate
before anything risky happens."""

from __future__ import annotations

import os
import sys

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_COLOR = _supports_color()

if os.name == "nt" and _COLOR:
    # Enable ANSI/VT100 escape processing on legacy Windows consoles (a
    # no-op on modern Windows Terminal, harmless if it fails silently).
    try:
        os.system("")
    except Exception:
        pass


def _c(text: str, *styles: str) -> str:
    if not _COLOR:
        return text
    prefix = "".join(_ANSI[s] for s in styles if s in _ANSI)
    return f"{prefix}{text}{_ANSI['reset']}"


def _rule(char: str = "-", width: int = 70) -> None:
    print(_c(char * width, "dim"))


def banner(root: str, model: str, max_ctx: int, memory_path: str) -> None:
    _rule("=")
    print(_c("harness", "bold", "cyan") + " - sandboxed local coding agent")
    print(f"  root:   {root}")
    print(f"  model:  {model}  (context {max_ctx} tokens)")
    print(f"  memory: {memory_path}")
    _rule("=")


def step_header(n: int, max_steps: int) -> None:
    _rule()
    print(_c(f"step {n}/{max_steps}", "bold"))


def thought(text: str) -> None:
    if text:
        print(_c(f"thought: {text}", "dim", "italic"))


def tool_call(name: str, args: dict) -> None:
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(_c(f"-> {name}", "bold", "yellow") + f"({args_str})")


def tool_result(ok: bool, output: str, limit: int = 1200) -> None:
    color = "green" if ok else "red"
    label = "result" if ok else "error"
    shown = output if len(output) <= limit else output[:limit] + f"\n... [truncated, {len(output)} chars]"
    print(_c(f"[{label}]", "bold", color))
    for line in shown.splitlines() or [""]:
        print(f"  {line}")


def diff(text: str) -> None:
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            style = "bold"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        else:
            style = "dim"
        print(_c(line, style))


def final_answer(text: str) -> None:
    _rule("=")
    print(_c("done", "bold", "cyan"))
    print(text)
    _rule("=")


def warn(text: str) -> None:
    print(_c(f"! {text}", "bold", "red"))


def info(text: str) -> None:
    print(_c(text, "dim"))


def confirm(prompt: str) -> bool:
    try:
        ans = input(_c(f"{prompt} [y/N] ", "bold", "magenta")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")
