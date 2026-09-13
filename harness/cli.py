from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ui
from .agent import Agent
from .config import Config
from .llm_client import LLMClient, detect_server_info
from .memory import MemoryStore
from .planner import Planner
from .sandbox import Sandbox, SandboxViolation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="Sandboxed, Antigravity-style coding agent.")
    p.add_argument("task", nargs="?", help="Task to run once. Omit to start an interactive session.")
    p.add_argument("--root", default=".", help="Sandbox root directory (default: current directory). "
                                                "All file/command access is confined to this directory.")
    p.add_argument("--base-url", dest="server_base_url", help="Override the LLM server base URL.")
    p.add_argument("--api-key", dest="api_key", help="Override the API key sent to the LLM server.")
    p.add_argument("--model", help="Override the model name (default: auto-detect).")
    p.add_argument("--max-steps", type=int, dest="max_steps", help="Max agent loop steps per task.")
    p.add_argument("--auto", action="store_true", help="Skip confirmation prompts (auto-approve all tools).")
    p.add_argument("--no-memory", action="store_true", help="Disable automatic memory recall at task start.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        root = Path(args.root).resolve(strict=True)
        sandbox = Sandbox(root)
    except (FileNotFoundError, SandboxViolation) as e:
        print(f"harness: invalid sandbox root: {e}", file=sys.stderr)
        return 1

    overrides = {
        "server_base_url": args.server_base_url,
        "api_key": args.api_key,
        "model": args.model,
        "max_steps": args.max_steps,
        "auto_approve": True if args.auto else None,
        "memory_auto_recall": False if args.no_memory else None,
    }
    try:
        config = Config.load(root, overrides)
    except ValueError as e:
        print(f"harness: {e}", file=sys.stderr)
        return 1

    info = detect_server_info(config)
    client = LLMClient(config, info.model)

    state_dir = sandbox.root / ".harness"
    memory = MemoryStore(state_dir / "memory.sqlite3", dim=config.memory_embedding_dim)
    planner = Planner(sandbox.root / "PLAN.md")
    agent = Agent(sandbox, config, client, memory, planner, info.max_context_tokens)

    ui.banner(str(sandbox.root), info.model, info.max_context_tokens, str(memory.db_path))
    if config.auto_approve:
        ui.warn("Running with --auto: file writes and commands will NOT ask for confirmation.")

    try:
        if args.task:
            agent.run_task(args.task)
            return 0

        ui.info("Interactive mode. Type a task, or 'exit'/'quit' to leave.")
        while True:
            try:
                task = input("\nharness> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not task:
                continue
            if task.lower() in ("exit", "quit"):
                break
            agent.run_task(task)
        return 0
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
