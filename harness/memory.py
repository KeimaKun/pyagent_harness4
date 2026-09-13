"""Embedded RAG memory: sqlite3 + a pure-Python hashed-bag-of-words embedding.

No compiled vector extension (e.g. sqlite-vec) and no ML embedding model
are required - both would be an extra thing to install, which this project
deliberately avoids. Instead, text is embedded with a deterministic
feature-hashing trick into a fixed-size float vector, and cosine
similarity is computed in plain Python over rows read back from sqlite3.
This is a lightweight, dependency-free approximation of semantic search:
good enough for an agent's own session notes and indexed project files at
the scale a local coding-agent session produces, not a replacement for a
real embedding model on a large corpus. A keyword-overlap score is blended
in alongside the cosine score so exact-term matches are never missed, as a
fallback/complement to the vector search (mirroring the "vector search
with keyword-search fallback" shape of a sqlite-vec-backed RAG store).

Storage lives in a single sqlite3 database (default:
``<sandbox root>/.harness/memory.sqlite3``), in one table:

    memory(id, source, content, embedding BLOB, dim, created_at)

`embedding` is a little-endian packed float array (via `struct`), the same
on-disk shape sqlite-vec itself uses, so a real vector extension could be
swapped in later without changing the table layout.
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
import time
import zlib
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _stable_hash(token: str) -> int:
    """A hash that is stable across processes/runs (unlike built-in `hash()`
    on str, which is salted per-process by PYTHONHASHSEED) - required since
    embeddings written in one run must compare correctly against query
    embeddings computed in a later run."""
    return zlib.crc32(token.encode("utf-8"))


def hash_embed(text: str, dim: int = 256) -> list[float]:
    """Feature-hashing embedding: each token votes +-1 into hash(token) % dim,
    then the vector is L2-normalized. Cheap, deterministic, no model needed."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = _stable_hash(tok)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks for indexing."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    step = max(1, chunk_chars - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


class MemoryStore:
    def __init__(self, db_path: str | Path, dim: int = 256):
        self.db_path = str(db_path)
        self.dim = dim
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, content: str, source: str = "note") -> int:
        content = content.strip()
        if not content:
            raise ValueError("Cannot store empty content in memory.")
        emb = hash_embed(content, self.dim)
        blob = struct.pack(f"{self.dim}f", *emb)
        cur = self.conn.execute(
            "INSERT INTO memory (source, content, embedding, dim, created_at) VALUES (?, ?, ?, ?, ?)",
            (source, content, blob, self.dim, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_chunks(self, text: str, source: str, chunk_chars: int = 1200) -> int:
        """Chunk `text` and store each chunk as its own memory row. Returns
        the number of chunks stored."""
        n = 0
        for i, chunk in enumerate(chunk_text(text, chunk_chars=chunk_chars)):
            self.add(chunk, source=f"{source}#chunk{i}")
            n += 1
        return n

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM memory").fetchone()
        return int(row["n"]) if row else 0

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Hybrid search: 70% cosine similarity on the hashed embedding,
        30% fraction of query tokens present in the candidate's text."""
        query = query.strip()
        if not query:
            return []
        query_vec = hash_embed(query, self.dim)
        query_tokens = set(_tokenize(query))

        rows = self.conn.execute("SELECT id, source, content, embedding, dim FROM memory").fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if row["dim"] != self.dim:
                continue  # embedding dimension changed since this row was written; skip it
            vec = struct.unpack(f"{row['dim']}f", row["embedding"])
            cos = sum(a * b for a, b in zip(query_vec, vec))  # both sides are L2-normalized
            content_tokens = set(_tokenize(row["content"]))
            overlap = len(query_tokens & content_tokens)
            kw_score = overlap / max(1, len(query_tokens))
            score = 0.7 * cos + 0.3 * kw_score
            scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "content": row["content"],
                "score": round(score, 4),
            }
            for score, row in scored[:top_k]
            if score > 0
        ]

    def close(self) -> None:
        self.conn.close()
