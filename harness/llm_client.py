"""Thin client around an OpenAI-API-compatible chat endpoint.

Built on `urllib.request` from the standard library rather than the
`requests` or `openai` packages, so this harness has zero third-party
dependencies - it runs under any Python 3.10+ interpreter, not just one
where those happen to be pip-installed. Talking plain HTTP also means it
works against any OpenAI-compatible server (llama.cpp server, LM Studio,
vLLM, text-generation-webui, a hosted API gateway, ...) without depending
on an SDK's own version quirks.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Config


@dataclass
class ServerInfo:
    model: str
    max_context_tokens: int


class LLMError(Exception):
    pass


def _get_json(url: str, timeout: float = 5.0) -> dict | None:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def detect_server_info(config: Config) -> ServerInfo:
    """Best-effort auto-detection of model name and context window."""
    model = config.model
    max_ctx = config.max_context_tokens
    base = config.server_base_url.rstrip("/")

    if model is None:
        data = _get_json(f"{base}/models")
        if data and data.get("data"):
            model = data["data"][0].get("id")

    if max_ctx is None:
        # llama.cpp server exposes /health with n_ctx at the server root,
        # one level up from the /v1 API prefix.
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        data = _get_json(f"{root}/health")
        if data and isinstance(data.get("n_ctx"), int):
            max_ctx = data["n_ctx"]

    return ServerInfo(
        model=model or "local-model",
        max_context_tokens=max_ctx or config.fallback_context_tokens,
    )


class LLMClient:
    def __init__(self, config: Config, model: str):
        self.config = config
        self.model = model
        self.base = config.server_base_url.rstrip("/")

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.reserved_completion_tokens,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=data, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout_sec) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise LLMError(f"LLM server returned HTTP {e.code}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Could not reach LLM server at {self.base}: {e.reason}") from e
        except OSError as e:
            raise LLMError(f"Could not reach LLM server at {self.base}: {e}") from e

        try:
            obj = json.loads(body)
            return obj["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response shape from LLM server: {e} (body: {body[:500]})") from e
