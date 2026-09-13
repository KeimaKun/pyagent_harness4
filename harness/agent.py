"""The agent loop: think -> act -> observe -> reflect, with a confirmation
gate on risky tools, an anti-repetition guard, and a final markdown
artifact per task - echoing Antigravity's plan/execute/summarize UX in a
plain terminal."""

from __future__ import annotations

from datetime import datetime, timezone

from . import ui
from .config import Config
from .llm_client import LLMClient
from .memory import MemoryStore
from .planner import Planner, RepeatGuard
from .protocol import parse_action
from .sandbox import Sandbox
from .tools import REGISTRY, Context, dispatch, unified_diff

SYSTEM_PROMPT_TEMPLATE = """You are a careful coding agent working ONLY inside the project directory below. \
You cannot see or touch anything outside it - all paths you use must be relative to it.

Project root: {root}

You solve the user's task by repeatedly choosing ONE tool to run, observing its result, and continuing \
until done. Respond with NOTHING but a single fenced block like this, no other text:

```action
{{"thought": "one short sentence", "tool": "<tool name>", "args": {{...}}}}
```

When the task is complete, respond the same way but with "final" instead of "tool":

```action
{{"thought": "one short sentence", "final": "short summary of what you did for the user"}}
```

Available tools:
{tool_docs}

Rules:
- Exactly one action per response. Never call more than one tool at a time.
- Always use paths relative to the project root.
- Prefer read_file/list_dir/grep_search/recall to look before you write.
- Use update_plan to keep a running plan for any task that takes more than a couple of steps.
- Use remember to save durable facts/decisions, and recall to check memory before re-exploring
  something you (or a past session) may have already found.
- If a tool result shows an error, fix your approach on the next step rather than repeating the
  same call unchanged.
- Keep "thought" to one short sentence - do not repeat the whole plan every step.
"""


def build_system_prompt(root: str) -> str:
    from .tools import TOOL_DOCS
    return SYSTEM_PROMPT_TEMPLATE.format(root=root, tool_docs=TOOL_DOCS)


def _char_budget(config: Config, max_context_tokens: int) -> int:
    usable_tokens = max(512, max_context_tokens - config.reserved_completion_tokens)
    return int(usable_tokens * config.chars_per_token)


def _msg_len(m: dict) -> int:
    return len(m.get("content", ""))


def trim_history(messages: list[dict], budget_chars: int) -> list[dict]:
    """Keep messages[0] (system) and messages[1] (initial task) always;
    drop oldest exchanges after that while over budget."""
    if len(messages) <= 2:
        return messages
    head, rest = messages[:2], messages[2:]
    total = sum(_msg_len(m) for m in messages)
    while total > budget_chars and len(rest) > 2:
        removed = rest.pop(0)
        total -= _msg_len(removed)
    return head + rest


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} chars total]"


class Agent:
    def __init__(self, sandbox: Sandbox, config: Config, client: LLMClient,
                 memory: MemoryStore, planner: Planner, max_context_tokens: int):
        self.sandbox = sandbox
        self.config = config
        self.client = client
        self.memory = memory
        self.planner = planner
        self.max_context_tokens = max_context_tokens
        self.artifacts_dir = sandbox.root / ".harness" / "artifacts"
        self.ctx = Context(sandbox=sandbox, config=config, planner=planner, memory=memory)
        self.repeat_guard = RepeatGuard(warn_limit=config.repeat_warn_limit)

    def _needs_confirmation(self, tool: str) -> bool:
        if self.config.auto_approve:
            return False
        return tool in self.config.confirm_tools

    def _confirm_tool(self, tool: str, args: dict) -> bool:
        if tool == "write_file":
            path, content = args.get("path", "?"), str(args.get("content", ""))
            try:
                target = self.sandbox.safe_path(path)
                old = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
            except Exception:
                old = ""
            ui.diff(unified_diff(old, content, path))
            return ui.confirm(f"Apply write to {path!r}?")
        if tool == "edit_file":
            path = args.get("path", "?")
            old_s, new_s = str(args.get("old_string", "")), str(args.get("new_string", ""))
            try:
                target = self.sandbox.safe_path(path)
                old_full = target.read_text(encoding="utf-8", errors="replace")
                new_full = old_full.replace(old_s, new_s, -1 if args.get("replace_all") else 1)
                ui.diff(unified_diff(old_full, new_full, path))
            except Exception:
                ui.info(f"old: {old_s!r}\nnew: {new_s!r}")
            return ui.confirm(f"Apply edit to {path!r}?")
        if tool == "delete_file":
            return ui.confirm(f"Delete {args.get('path', '?')!r}? This cannot be undone.")
        if tool in ("run_command", "run_python"):
            shown = args.get("command") or args.get("path") or args.get("code", "")
            ui.info(f"about to run in {self.sandbox.root}:\n{shown}")
            return ui.confirm("Run this?")
        return ui.confirm(f"Run {tool}({args})?")

    def _auto_recall_block(self, task: str) -> str:
        if not self.config.memory_auto_recall or self.memory.count() == 0:
            return ""
        hits = self.memory.search(task, top_k=self.config.memory_top_k)
        if not hits:
            return ""
        lines = [f"- [{h['score']:.2f}] ({h['source']}) {h['content']}" for h in hits]
        return "\n\nRelevant memory (from previous sessions, via recall):\n" + "\n".join(lines)

    def run_task(self, task: str) -> str:
        system_prompt = build_system_prompt(str(self.sandbox.root))
        user_content = f"Task: {task}" + self._auto_recall_block(task)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        budget = _char_budget(self.config, self.max_context_tokens)
        transcript: list[str] = []
        invalid_retries = 0

        for step in range(1, self.config.max_steps + 1):
            ui.step_header(step, self.config.max_steps)
            messages = trim_history(messages, budget)

            try:
                reply = self.client.chat(messages)
            except Exception as e:  # noqa: BLE001
                ui.warn(f"LLM request failed: {e}")
                return f"Stopped: LLM request failed ({e})"

            action = parse_action(reply)
            messages.append({"role": "assistant", "content": reply})
            ui.thought(action.thought)

            if action.kind == "invalid":
                invalid_retries += 1
                ui.warn(f"Could not parse a valid action: {action.error}")
                if invalid_retries > 2:
                    ui.warn("Giving up after repeated invalid responses.")
                    return reply.strip() or "Stopped: model did not produce a usable action."
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your last response was not valid: {action.error} "
                        "Reply with ONLY a single ```action fenced JSON block as instructed."
                    ),
                })
                continue

            invalid_retries = 0

            if action.kind == "final":
                ui.final_answer(action.final_text or "(no summary provided)")
                self._write_artifact(task, transcript, action.final_text or "")
                return action.final_text or ""

            # action.kind == "tool"
            tool, args = action.tool, action.args or {}
            ui.tool_call(tool, args)

            streak = self.repeat_guard.record_and_check(tool, args)
            if streak >= self.config.repeat_abort_limit:
                ui.warn(f"Aborting: {tool} called identically {streak} times in a row (stuck loop).")
                self._write_artifact(task, transcript, "(stopped: repeated identical tool call, likely stuck)")
                return "Stopped: the agent got stuck repeating the same tool call and was aborted."
            if streak >= self.config.repeat_warn_limit:
                result = {"ok": False, "error": (
                    f"You have now called {tool} with these exact arguments {streak} times in a row. "
                    "This is not being executed again - change your approach, use different arguments, "
                    "or finish with a 'final' response."
                )}
                ok = False
                output = result["error"]
            elif tool not in REGISTRY:
                ok, output = False, f"Unknown tool {tool!r}. Available: {', '.join(REGISTRY)}"
            elif self._needs_confirmation(tool) and not self._confirm_tool(tool, args):
                ok, output = False, "User declined to run this action."
            else:
                result = dispatch(tool, self.ctx, args)
                ok = bool(result.get("ok"))
                output = result.get("output") if ok else result.get("error", "unknown error")

            ui.tool_result(ok, output)
            transcript.append(f"### Step {step}: {tool}({args})\n{'OK' if ok else 'ERROR'}: {output}\n")

            observation = _truncate(output, self.config.tool_output_char_limit)
            messages.append({
                "role": "user",
                "content": f"Tool `{tool}` result ({'ok' if ok else 'error'}):\n{observation}",
            })

        ui.warn(f"Reached max_steps ({self.config.max_steps}) without a final answer.")
        self._write_artifact(task, transcript, "(stopped: max steps reached)")
        return "Stopped: reached max_steps without finishing."

    def _write_artifact(self, task: str, transcript: list[str], summary: str) -> None:
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = self.artifacts_dir / f"{ts}.md"
            body = (
                f"# Task\n{task}\n\n"
                f"# Summary\n{summary}\n\n"
                f"# Trace\n" + "\n".join(transcript)
            )
            path.write_text(body, encoding="utf-8")
            ui.info(f"artifact written: {self.sandbox.rel(path)}")
        except OSError as e:
            ui.warn(f"Could not write artifact: {e}")
