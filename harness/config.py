"""Configuration loading for `harness`.

Precedence (low -> high): built-in defaults < harness.config.json in the
sandbox root < CLI flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "harness.config.json"

DEFAULTS: dict[str, Any] = {
    # Any OpenAI-API-compatible chat endpoint: a local llama.cpp server,
    # LM Studio, vLLM, text-generation-webui, or a hosted provider.
    "server_base_url": "http://127.0.0.1:8000/v1",
    "api_key": "not-needed",
    # None => auto-detect from GET /v1/models at startup.
    "model": None,
    # None => auto-detect from GET /health ("n_ctx") at startup; falls
    # back to fallback_context_tokens if detection fails.
    "max_context_tokens": None,
    "fallback_context_tokens": 4096,
    "reserved_completion_tokens": 512,
    "temperature": 0.2,
    "max_steps": 25,
    "command_timeout_sec": 120,
    "request_timeout_sec": 180,
    # Rough chars-per-token used to budget history size.
    "chars_per_token": 3.3,
    "tool_output_char_limit": 2500,
    # Tools listed here always ask for interactive y/n confirmation before
    # running, regardless of --auto (only --auto's explicit override, or
    # running non-interactively, changes this - see agent.py).
    "confirm_tools": ["write_file", "edit_file", "delete_file", "run_command", "run_python"],
    "auto_approve": False,
    # None => resolved at load time (see Config.ml_env_python).
    "path_to_ml_env_python": None,
    # Memory / RAG (harness/memory.py) knobs.
    "memory_top_k": 5,
    "memory_embedding_dim": 256,
    "memory_chunk_chars": 1200,
    "memory_auto_recall": True,
    # Anti-repetition guard: how many identical consecutive tool calls
    # (same tool + same args) are tolerated before the agent is nudged,
    # and how many more after that before the task is aborted as stuck.
    "repeat_warn_limit": 3,
    "repeat_abort_limit": 6,
}


@dataclass
class Config:
    server_base_url: str
    api_key: str
    model: str | None
    max_context_tokens: int | None
    fallback_context_tokens: int
    reserved_completion_tokens: int
    temperature: float
    max_steps: int
    command_timeout_sec: int
    request_timeout_sec: int
    chars_per_token: float
    tool_output_char_limit: int
    confirm_tools: list[str]
    auto_approve: bool
    path_to_ml_env_python: str | None
    memory_top_k: int
    memory_embedding_dim: int
    memory_chunk_chars: int
    memory_auto_recall: bool
    repeat_warn_limit: int
    repeat_abort_limit: int

    @classmethod
    def load(cls, root: Path, overrides: dict[str, Any] | None = None) -> "Config":
        data = dict(DEFAULTS)

        config_file = root / CONFIG_FILENAME
        if config_file.exists():
            try:
                user_cfg = json.loads(config_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                raise ValueError(f"Failed to read {config_file}: {e}") from e
            if not isinstance(user_cfg, dict):
                raise ValueError(f"{config_file} must contain a JSON object.")
            data.update({k: v for k, v in user_cfg.items() if k in DEFAULTS})

        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None and k in DEFAULTS})

        valid_names = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_names}
        return cls(**data)

    def ml_env_python(self, root: Path) -> str | None:
        """Resolve the interpreter used by the run_python tool.

        Search order: explicit config value -> <sandbox root>/python_path.txt
        -> the harness package's own installation directory's
        python_path.txt (lets run_python work out of the box when the
        sandbox root is some *other* project directory, as long as this
        same harness install has one next to it).
        """
        if self.path_to_ml_env_python:
            return self.path_to_ml_env_python

        for candidate_dir in (root, Path(__file__).resolve().parent.parent):
            candidate = candidate_dir / "python_path.txt"
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text
        return None
