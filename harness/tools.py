"""Tool implementations available to the agent.

Every tool function takes (ctx: Context, **args) and returns a plain dict:
{"ok": True, "output": <str>} on success or {"ok": False, "error": <str>}
on failure. Nothing here should ever raise for "expected" failures (file
not found, bad args, etc.) - only truly unexpected errors propagate, and
even those are caught by `dispatch` below so the agent loop never crashes.
"""

from __future__ import annotations

import difflib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .memory import MemoryStore, chunk_text
from .planner import Planner
from .sandbox import Sandbox, SandboxViolation

MAX_READ_CHARS = 8000
IGNORED_DIR_NAMES = {".git", "__pycache__", ".harness", "node_modules", ".venv", "venv",
                     ".mypy_cache", ".pytest_cache"}


@dataclass
class Context:
    """Bundles everything a tool call needs: the sandbox boundary, config,
    and the stateful plan/memory stores shared across a whole session."""

    sandbox: Sandbox
    config: Config
    planner: Planner
    memory: MemoryStore


def _ok(output: str) -> dict[str, Any]:
    return {"ok": True, "output": output}


def _err(msg: str) -> dict[str, Any]:
    return {"ok": False, "error": msg}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

def list_dir(ctx: Context, path: str = ".", **_: Any) -> dict[str, Any]:
    try:
        target = ctx.sandbox.safe_path(path, must_exist=True)
    except SandboxViolation as e:
        return _err(str(e))
    if not target.is_dir():
        return _err(f"{path!r} is not a directory")

    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.name in IGNORED_DIR_NAMES:
            continue
        kind = "dir" if child.is_dir() else "file"
        size = "" if kind == "dir" else f" ({child.stat().st_size}B)"
        entries.append(f"{kind:4}  {ctx.sandbox.rel(child)}{size}")
    if not entries:
        return _ok("(empty directory)")
    return _ok("\n".join(entries))


def read_file(ctx: Context, path: str, start_line: int | None = None,
              end_line: int | None = None, **_: Any) -> dict[str, Any]:
    try:
        target = ctx.sandbox.safe_path(path, must_exist=True)
    except SandboxViolation as e:
        return _err(str(e))
    if target.is_dir():
        return _err(f"{path!r} is a directory, use list_dir instead")

    try:
        text = _read_text(target)
    except OSError as e:
        return _err(f"Could not read {path!r}: {e}")

    lines = text.splitlines()
    if start_line or end_line:
        s = max(1, start_line or 1)
        e = min(len(lines), end_line or len(lines))
        numbered = "\n".join(f"{i:>5}: {lines[i - 1]}" for i in range(s, e + 1))
        return _ok(numbered)

    if len(text) > MAX_READ_CHARS:
        truncated = text[:MAX_READ_CHARS]
        return _ok(truncated + f"\n... [truncated, {len(text)} chars total; "
                                f"pass start_line/end_line to read a smaller range]")
    return _ok(text if text else "(empty file)")


def write_file(ctx: Context, path: str, content: str, **_: Any) -> dict[str, Any]:
    try:
        target = ctx.sandbox.safe_path(path)
    except SandboxViolation as e:
        return _err(str(e))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return _err(f"Could not write {path!r}: {e}")
    return _ok(f"Wrote {len(content)} chars to {ctx.sandbox.rel(target)}")


def edit_file(ctx: Context, path: str, old_string: str, new_string: str,
              replace_all: bool = False, **_: Any) -> dict[str, Any]:
    try:
        target = ctx.sandbox.safe_path(path, must_exist=True)
    except SandboxViolation as e:
        return _err(str(e))
    try:
        text = _read_text(target)
    except OSError as e:
        return _err(f"Could not read {path!r}: {e}")

    count = text.count(old_string)
    if count == 0:
        return _err("old_string not found in file (must match exactly, including whitespace).")
    if count > 1 and not replace_all:
        return _err(f"old_string is not unique ({count} occurrences). Pass replace_all=true "
                     "or include more surrounding context to make it unique.")

    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    try:
        target.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return _err(f"Could not write {path!r}: {e}")
    n = count if replace_all else 1
    return _ok(f"Replaced {n} occurrence(s) in {ctx.sandbox.rel(target)}")


def delete_file(ctx: Context, path: str, **_: Any) -> dict[str, Any]:
    try:
        target = ctx.sandbox.safe_path(path, must_exist=True)
    except SandboxViolation as e:
        return _err(str(e))
    try:
        if target.is_dir():
            return _err("delete_file only deletes files, not directories.")
        target.unlink()
    except OSError as e:
        return _err(f"Could not delete {path!r}: {e}")
    return _ok(f"Deleted {ctx.sandbox.rel(target)}")


def glob_search(ctx: Context, pattern: str, path: str = ".", **_: Any) -> dict[str, Any]:
    try:
        base = ctx.sandbox.safe_path(path, must_exist=True)
    except SandboxViolation as e:
        return _err(str(e))
    matches = []
    for p in base.rglob(pattern):
        if any(part in IGNORED_DIR_NAMES for part in p.parts):
            continue
        matches.append(ctx.sandbox.rel(p))
        if len(matches) >= 200:
            break
    if not matches:
        return _ok("(no matches)")
    return _ok("\n".join(sorted(matches)))


def grep_search(ctx: Context, pattern: str, path: str = ".",
                 glob: str | None = None, max_results: int = 50, **_: Any) -> dict[str, Any]:
    try:
        base = ctx.sandbox.safe_path(path, must_exist=True)
    except SandboxViolation as e:
        return _err(str(e))
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return _err(f"Invalid regex: {e}")

    results = []
    files_iter = base.rglob(glob) if glob else base.rglob("*")
    for f in files_iter:
        if not f.is_file() or any(part in IGNORED_DIR_NAMES for part in f.parts):
            continue
        try:
            text = _read_text(f)
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append(f"{ctx.sandbox.rel(f)}:{i}: {line.strip()[:200]}")
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break
    if not results:
        return _ok("(no matches)")
    return _ok("\n".join(results))


# ---------------------------------------------------------------------------
# Execution tools
# ---------------------------------------------------------------------------

def run_command(ctx: Context, command: str, timeout: int | None = None, **_: Any) -> dict[str, Any]:
    check = ctx.sandbox.check_command(command)
    if not check.ok:
        return _err(check.reason)

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.sandbox.root),
            capture_output=True,
            text=True,
            timeout=timeout or ctx.config.command_timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return _err(f"Command timed out after {timeout or ctx.config.command_timeout_sec}s")
    except OSError as e:
        return _err(f"Failed to run command: {e}")

    out = f"$ {command}\n[exit code {proc.returncode}]\n"
    if proc.stdout:
        out += f"--- stdout ---\n{proc.stdout}\n"
    if proc.stderr:
        out += f"--- stderr ---\n{proc.stderr}\n"
    return _ok(out.strip())


def run_python(ctx: Context, code: str | None = None, path: str | None = None,
               args: list[str] | None = None, timeout: int | None = None, **_: Any) -> dict[str, Any]:
    py = ctx.config.ml_env_python(ctx.sandbox.root)
    if not py:
        return _err("No Python interpreter configured (python_path.txt missing).")
    if not Path(py).exists():
        return _err(f"Configured Python interpreter does not exist: {py}")

    if code and path:
        return _err("Pass either code or path, not both.")
    if not code and not path:
        return _err("Must provide code or path.")

    if path:
        try:
            target = ctx.sandbox.safe_path(path, must_exist=True)
        except SandboxViolation as e:
            return _err(str(e))
        cmd = [py, str(target), *list(args or [])]
        label = f"{py} {ctx.sandbox.rel(target)}"
    else:
        cmd = [py, "-c", code]
        label = f"{py} -c <inline code>"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ctx.sandbox.root),
            capture_output=True,
            text=True,
            timeout=timeout or ctx.config.command_timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return _err(f"Python execution timed out after {timeout or ctx.config.command_timeout_sec}s")
    except OSError as e:
        return _err(f"Failed to run python: {e}")

    out = f"$ {label}\n[exit code {proc.returncode}]\n"
    if proc.stdout:
        out += f"--- stdout ---\n{proc.stdout}\n"
    if proc.stderr:
        out += f"--- stderr ---\n{proc.stderr}\n"
    return _ok(out.strip())


# ---------------------------------------------------------------------------
# Plan tool
# ---------------------------------------------------------------------------

def update_plan(ctx: Context, summary: str = "", current: str = "",
                 completed: list[str] | None = None, next_steps: list[str] | None = None,
                 **_: Any) -> dict[str, Any]:
    return ctx.planner.update(summary=summary, current=current, completed=completed, next_steps=next_steps)


# ---------------------------------------------------------------------------
# Memory / RAG tools
# ---------------------------------------------------------------------------

def remember(ctx: Context, content: str, source: str = "note", **_: Any) -> dict[str, Any]:
    try:
        row_id = ctx.memory.add(content, source=source)
    except ValueError as e:
        return _err(str(e))
    return _ok(f"Stored memory #{row_id} (source={source!r}).")


def recall(ctx: Context, query: str, top_k: int | None = None, **_: Any) -> dict[str, Any]:
    results = ctx.memory.search(query, top_k=top_k or ctx.config.memory_top_k)
    if not results:
        return _ok("(no relevant memory found)")
    lines = [f"[{r['score']:.2f}] ({r['source']}) {r['content']}" for r in results]
    return _ok("\n".join(lines))


def index_file(ctx: Context, path: str, **_: Any) -> dict[str, Any]:
    try:
        target = ctx.sandbox.safe_path(path, must_exist=True)
    except SandboxViolation as e:
        return _err(str(e))
    if target.is_dir():
        return _err(f"{path!r} is a directory, index files individually.")
    try:
        text = _read_text(target)
    except OSError as e:
        return _err(f"Could not read {path!r}: {e}")
    rel = ctx.sandbox.rel(target)
    n = ctx.memory.add_chunks(text, source=rel, chunk_chars=ctx.config.memory_chunk_chars)
    return _ok(f"Indexed {n} chunk(s) from {rel} into memory.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ToolFunc = Callable[..., dict[str, Any]]

# Compact docs embedded into the system prompt. Kept short deliberately -
# some local models only have a small context window.
TOOL_DOCS = """\
File & search:
- list_dir(path="."): list files/dirs under path.
- read_file(path, start_line=None, end_line=None): read a text file, optionally a line range.
- write_file(path, content): create/overwrite a file with content (full file text).
- edit_file(path, old_string, new_string, replace_all=false): replace exact text in a file.
- delete_file(path): delete a file.
- glob_search(pattern, path="."): find files by glob pattern, e.g. "*.py".
- grep_search(pattern, path=".", glob=None, max_results=50): regex search file contents.

Execution:
- run_command(command, timeout=None): run a shell command in the project root.
- run_python(code=None, path=None, args=None, timeout=None): run Python via the configured interpreter.

Planning:
- update_plan(summary, current, completed=[...], next_steps=[...]): overwrite the plan file with the
  current state of the task. Use it for any multi-step task so progress survives context trimming.

Memory (RAG, backed by sqlite3):
- remember(content, source="note"): permanently store an important fact/decision for later recall,
  in this and future sessions against the same project.
- recall(query, top_k=5): semantic + keyword search over everything stored with remember/index_file.
- index_file(path): chunk a file's contents into memory so recall() can search it later.\
"""

REGISTRY: dict[str, ToolFunc] = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "glob_search": glob_search,
    "grep_search": grep_search,
    "run_command": run_command,
    "run_python": run_python,
    "update_plan": update_plan,
    "remember": remember,
    "recall": recall,
    "index_file": index_file,
}


def dispatch(name: str, ctx: Context, args: dict[str, Any]) -> dict[str, Any]:
    func = REGISTRY.get(name)
    if func is None:
        return _err(f"Unknown tool {name!r}. Available tools: {', '.join(REGISTRY)}")
    try:
        return func(ctx, **args)
    except TypeError as e:
        return _err(f"Bad arguments for {name}: {e}")
    except Exception as e:  # noqa: BLE001 - last-resort guard so the loop never crashes
        return _err(f"Tool {name} raised an unexpected error: {e}")


def unified_diff(old: str, new: str, label: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{label} (before)",
        tofile=f"{label} (after)",
    )
    text = "".join(diff)
    return text if text else "(no textual difference)"
