# harness

A small, sandboxed, terminal coding-agent CLI styled loosely after **Google
Antigravity**'s agent-loop UX: plan → act → observe, a visible step trace,
a confirmation gate before risky actions, a running plan file, and a final
markdown "artifact" per task. It talks to any OpenAI-API-compatible chat
endpoint (a local `llama.cpp`/LM Studio/vLLM server, or a hosted one).

This is `pyagent-harness4`: a from-scratch rebuild that folds together the
better ideas from the two reference harnesses kept alongside it in this
project —

- **[`pyagent-harness2/`](pyagent-harness2/)** — native OpenAI
  function-calling, a `Planner` that atomically overwrites `PLAN.md`
  (summary / current / completed / next), and a `MemoryStore` backed by
  `sqlite3` with an optional `sqlite-vec` extension for vector search.
- **[`pyagent_harness3/`](pyagent_harness3/)** — a prompted (non-native)
  JSON tool-call protocol robust enough for small local models, hard
  path-containment sandboxing, and honest documentation of exactly what
  is and isn't actually enforced.

`harness/` (this project's package) takes harness3's protocol/sandbox
design as its backbone, adds harness2's structured plan format and a
`sqlite3`-backed memory store, and — the point of rebuilding it — drops
**every third-party dependency**. The whole thing is Python standard
library only (`sqlite3`, `urllib`, `argparse`, `subprocess`, `pathlib`,
`difflib`, ...). Nothing needs to be `pip install`ed to run it, on this
machine or any other Python 3.10+ install.

## Why zero dependencies

Both reference harnesses need packages installed: harness2 needs `openai`,
`sqlite-vec`, `httpx`; harness3 needs `requests` and `rich` (both of which
happen to already be present in this machine's `ml-env`, see
[`ml-env_requirements.txt`](ml-env_requirements.txt), but that's incidental
— point it at a fresh interpreter and it breaks). This rebuild removes that
fragility entirely: `urllib.request` replaces `requests`/`openai` for talking
to the model server, and a plain `sqlite3` table plus a deterministic
Python hashing function replaces the `sqlite-vec` compiled extension for
vector search. Terminal rendering uses raw ANSI codes instead of `rich`.

## Sandboxing — what's actually enforced

Everything is confined to one **sandbox root**: the directory you launch
`harness` from, or `--root`. **The agent cannot read, write, or execute
anything outside that directory.** Two different guarantees are made, and
they are **not** equally strong — see the docstring in
[`harness/sandbox.py`](harness/sandbox.py) for detail:

1. **File tools** (`read_file`, `write_file`, `edit_file`, `delete_file`,
   `list_dir`, `glob_search`, `grep_search`, `index_file`) **are hard-contained.**
   Every path is resolved (symlinks followed, `..` normalized) and checked
   against the sandbox root; anything that resolves outside it is rejected.
   This cannot be bypassed by the model.

2. **Shell/Python execution** (`run_command`, `run_python`) **is best-effort
   only.** A real OS-level jail (containers, a restricted token/AppContainer,
   a VM) is out of scope for this lightweight CLI tool. What's actually done:
   the subprocess's `cwd` is always pinned to the sandbox root, and the
   command string is statically rejected if it contains obvious escape
   patterns (`..\`/`../`, absolute paths outside the root, drive switches,
   UNC paths, `%ENV%`/`$ENV` expansion, `~`) or names a "living off the
   land" / system-level binary (`shutdown`, `reg`, `netsh`, `format`,
   `certutil`, `mshta`, `regsvr32`, etc.). **A sufficiently adversarial
   command could still defeat this** — treat it as "won't wander outside
   by accident," not "cannot escape on purpose."

## Confirmation gate

By default, `write_file`, `edit_file`, `delete_file`, `run_command`, and
`run_python` all pause and show a diff (for file edits) or the exact
command (for execution) before running, and ask `y/n`. Every other tool
(reads, plan updates, memory) never needs confirmation. Pass `--auto` to
skip all confirmations (fully autonomous — use with care).

## Plan file

`update_plan(summary, current, completed=[...], next_steps=[...])`
atomically overwrites `PLAN.md` in the sandbox root with the latest state
(never an append-only log, so it can't bloat the model's context). Writing
an identical plan twice is a no-op. See [`harness/planner.py`](harness/planner.py).

## Memory (RAG via `sqlite3`)

`harness/memory.py` implements a small embedded RAG store using only
`sqlite3` plus a deterministic feature-hashing embedding written in plain
Python — no compiled vector extension, no embedding model/API call
required:

- `remember(content, source="note")` — permanently store a fact/decision.
- `recall(query, top_k=5)` — hybrid search (70% cosine similarity on the
  hashed embedding + 30% keyword overlap) over everything stored.
- `index_file(path)` — chunk a file's contents into memory so `recall` can
  search it later.

The database lives at `<sandbox root>/.harness/memory.sqlite3` and
persists across sessions against the same project. At the start of every
task, the agent automatically runs `recall(task)` and, if it finds
anything relevant, includes it as context — this is the "RAG" loop: what
gets `remember`ed or `index_file`d in one session becomes retrievable
grounding in the next. Disable the automatic step with `--no-memory`
(the `remember`/`recall`/`index_file` tools stay available either way).

## Anti-repetition guard

`harness/planner.py`'s `RepeatGuard` tracks the last several (tool, args)
calls. If the model calls the exact same tool with the exact same
arguments 3 times in a row, the 3rd+ call is refused with a message
telling it to change approach instead of being executed again; after 6
in a row the task is aborted outright as a stuck loop.

## Running it

```powershell
.\run_agent.ps1 "list every python file in this project and summarize what each does"
```

or with no task, for an interactive session:

```powershell
.\run_agent.ps1
```

`run_agent.ps1` resolves the interpreter from [`python_path.txt`](python_path.txt)
(this machine's `ml-env` conda environment — see
[`ml-env_requirements.txt`](ml-env_requirements.txt) for what's installed
there; `run_python` uses this same interpreter to execute code). Since the
harness itself has no third-party dependencies, you can equally run it
under any Python 3.10+:

```powershell
python -m harness
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--root PATH` | Sandbox root (default: current directory) |
| `--base-url URL` | Override the LLM server's base URL (default `http://127.0.0.1:8000/v1`) |
| `--api-key KEY` | Override the API key sent to the server |
| `--model NAME` | Override model name (default: auto-detected from `/v1/models`) |
| `--max-steps N` | Cap agent loop steps per task (default 25) |
| `--auto` | Skip all confirmation prompts |
| `--no-memory` | Disable automatic memory recall at task start |

## Configuration file

Optional [`harness.config.json`](harness.config.json) in the sandbox root
(a starter one is committed here) overrides defaults; CLI flags override
the config file. See [`harness/config.py`](harness/config.py) for every
field (timeouts, context-budget knobs, confirm-tools list, memory/RAG
knobs, repeat-guard limits, etc.).

## Artifacts

Each completed (or max-steps-exhausted) task writes a markdown summary —
task, final answer, full step trace — to `.harness/artifacts/<timestamp>.md`
inside the sandbox root, similar to Antigravity's per-task artifact.

## Layout

```
harness/
  sandbox.py      hard path containment + best-effort command filtering
  config.py       defaults + harness.config.json + CLI overrides
  memory.py       sqlite3-backed RAG memory (hashed embeddings + keyword blend)
  planner.py      PLAN.md overwrite-state planner + anti-repetition guard
  tools.py        tool implementations (file, exec, plan, memory) + registry
  protocol.py     parses the model's fenced-JSON action/final replies
  llm_client.py   stdlib-urllib client for any OpenAI-compatible server
  agent.py        the think/act/observe loop, history trimming, artifacts
  ui.py           stdlib ANSI terminal rendering (steps, diffs, confirms)
  cli.py          argparse entrypoint, one-shot or interactive REPL
tests/            unittest suite (stdlib `unittest`, no pytest needed)
run_agent.ps1     launcher that resolves the interpreter from python_path.txt
harness.config.json
```

## Testing

Uses only the standard library `unittest` module (no `pytest` install
required):

```powershell
python -m unittest discover -s tests -v
```

## Known limitations

- No native tool-calling — the prompted JSON-block protocol is used
  unconditionally, which works against both small local models and hosted
  ones, at the cost of relying on the model reliably following the format
  (parsing tolerates minor slips; the agent asks the model to retry a
  couple of times before giving up).
- `run_command`/`run_python` sandboxing is static-pattern-based, not a
  real OS jail (see above) — don't run this against untrusted task
  prompts expecting hard isolation.
- The RAG memory's hashed embeddings are a lightweight bag-of-words
  approximation of semantic similarity (feature hashing + cosine, blended
  with keyword overlap) — good for an agent's own notes and indexed
  project files at session scale, not a substitute for a real embedding
  model over a large corpus.
- Small context windows (common on local models) mean long files, long
  outputs, and long multi-step tasks get truncated/trimmed; the agent
  trades completeness for staying within budget.
