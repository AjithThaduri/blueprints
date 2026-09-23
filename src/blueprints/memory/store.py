"""Tiered agent memory on SQLite, from the "Agent memory without replaying
the whole conversation" blueprint.

Tiers
  pinned     short, always-in-prompt profile and rules, edited in place
  episodes   every turn, raw and append-only: the ground truth
  facts      extracted statements with validity windows; a changed fact
             invalidates the old one instead of overwriting it
  summary    rolling summary of turns older than the recent window

SQLite with FTS5 gives keyword search with no extra services. The same
schema maps directly onto Postgres (tsvector + pgvector) when you outgrow it.
Every read and write is scoped to a user_id; ``forget_user`` removes a user
from every tier.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS pinned (
    user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    updated_at REAL NOT NULL, PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS episodes_user ON episodes (user_id, id);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL,
    subject TEXT NOT NULL, predicate TEXT NOT NULL, value TEXT NOT NULL,
    source_episode INTEGER, valid_from REAL NOT NULL, valid_until REAL,
    created_at REAL NOT NULL, last_used REAL
);
CREATE INDEX IF NOT EXISTS facts_key ON facts (user_id, subject, predicate, valid_until);
CREATE TABLE IF NOT EXISTS summaries (
    user_id TEXT PRIMARY KEY, text TEXT NOT NULL, through_episode INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(content, content='episodes', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text);
"""


@dataclass
class Episode:
    id: int
    role: str
    content: str
    created_at: float


@dataclass
class Fact:
    id: int
    subject: str
    predicate: str
    value: str
    valid_from: float
    valid_until: float | None

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.value}"

    @property
    def current(self) -> bool:
        return self.valid_until is None


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms."""
    terms = [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(t) > 2]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms)) or '""'


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:", clock=time.time):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.clock = clock

    # ----------------------------------------------------------- pinned
    def pin(self, user_id: str, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO pinned VALUES (?, ?, ?, ?) ON CONFLICT(user_id, key) "
            "DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (user_id, key, value, self.clock()),
        )
        self.db.commit()

    def pinned(self, user_id: str) -> dict[str, str]:
        rows = self.db.execute("SELECT key, value FROM pinned WHERE user_id = ? ORDER BY key", (user_id,))
        return {r["key"]: r["value"] for r in rows}

    # --------------------------------------------------------- episodes
    def log(self, user_id: str, role: str, content: str) -> int:
        cur = self.db.execute(
            "INSERT INTO episodes (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, self.clock()),
        )
        self.db.execute("INSERT INTO episodes_fts (rowid, content) VALUES (?, ?)", (cur.lastrowid, content))
        self.db.commit()
        return cur.lastrowid

    def recent(self, user_id: str, n: int) -> list[Episode]:
        rows = self.db.execute(
            "SELECT id, role, content, created_at FROM episodes WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, n),
        ).fetchall()
        return [Episode(**dict(r)) for r in reversed(rows)]

    def search_episodes(self, user_id: str, query: str, k: int = 5, exclude_recent: int = 0) -> list[Episode]:
        recent_ids = {e.id for e in self.recent(user_id, exclude_recent)} if exclude_recent else set()
        rows = self.db.execute(
            "SELECT e.id, e.role, e.content, e.created_at FROM episodes_fts f "
            "JOIN episodes e ON e.id = f.rowid WHERE episodes_fts MATCH ? AND e.user_id = ? "
            "ORDER BY bm25(episodes_fts) LIMIT ?",
            (_fts_query(query), user_id, k + len(recent_ids)),
        ).fetchall()
        return [Episode(**dict(r)) for r in rows if r["id"] not in recent_ids][:k]

    # ------------------------------------------------------------ facts
    def add_fact(
        self, user_id: str, subject: str, predicate: str, value: str, source_episode: int | None = None
    ) -> Fact:
        """Add a fact. A current fact with the same subject and predicate but a
        different value is closed (valid_until = now), never deleted, so the
        history stays answerable ("what did we think last week?")."""
        now = self.clock()
        same = self.db.execute(
            "SELECT id, value FROM facts WHERE user_id = ? AND subject = ? AND predicate = ? AND valid_until IS NULL",
            (user_id, subject, predicate),
        ).fetchall()
        for row in same:
            if row["value"] == value:
                return self._fact(row["id"])  # already known; nothing to do
            self.db.execute("UPDATE facts SET valid_until = ? WHERE id = ?", (now, row["id"]))
        cur = self.db.execute(
            "INSERT INTO facts (user_id, subject, predicate, value, source_episode, valid_from, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, subject, predicate, value, source_episode, now, now),
        )
        self.db.execute(
            "INSERT INTO facts_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, f"{subject} {predicate} {value}")
        )
        self.db.commit()
        return self._fact(cur.lastrowid)

    def _fact(self, fact_id: int) -> Fact:
        r = self.db.execute(
            "SELECT id, subject, predicate, value, valid_from, valid_until FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return Fact(**dict(r))

    def facts(self, user_id: str, include_history: bool = False) -> list[Fact]:
        sql = "SELECT id, subject, predicate, value, valid_from, valid_until FROM facts WHERE user_id = ?"
        if not include_history:
            sql += " AND valid_until IS NULL"
        return [Fact(**dict(r)) for r in self.db.execute(sql + " ORDER BY id", (user_id,))]

    def search_facts(self, user_id: str, query: str, k: int = 8) -> list[Fact]:
        rows = self.db.execute(
            "SELECT f.id, f.subject, f.predicate, f.value, f.valid_from, f.valid_until "
            "FROM facts_fts s JOIN facts f ON f.id = s.rowid "
            "WHERE facts_fts MATCH ? AND f.user_id = ? AND f.valid_until IS NULL "
            "ORDER BY bm25(facts_fts) LIMIT ?",
            (_fts_query(query), user_id, k),
        ).fetchall()
        found = [Fact(**dict(r)) for r in rows]
        if found:
            self.db.executemany("UPDATE facts SET last_used = ? WHERE id = ?", [(self.clock(), f.id) for f in found])
            self.db.commit()
        return found

    def expire_unused(self, user_id: str, older_than_seconds: float) -> int:
        """Close facts nobody has retrieved for a long time. They stay in
        history but stop competing for the prompt budget."""
        cutoff = self.clock() - older_than_seconds
        cur = self.db.execute(
            "UPDATE facts SET valid_until = ? WHERE user_id = ? AND valid_until IS NULL "
            "AND COALESCE(last_used, created_at) < ?",
            (self.clock(), user_id, cutoff),
        )
        self.db.commit()
        return cur.rowcount

    # ---------------------------------------------------------- summary
    def summary(self, user_id: str) -> tuple[str, int]:
        r = self.db.execute("SELECT text, through_episode FROM summaries WHERE user_id = ?", (user_id,)).fetchone()
        return (r["text"], r["through_episode"]) if r else ("", 0)

    def set_summary(self, user_id: str, text: str, through_episode: int) -> None:
        self.db.execute(
            "INSERT INTO summaries VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
            "text = excluded.text, through_episode = excluded.through_episode",
            (user_id, text, through_episode),
        )
        self.db.commit()

    # ---------------------------------------------------------- privacy
    def forget_user(self, user_id: str) -> None:
        """Delete a user from every tier, including the search indexes."""
        episodes = self.db.execute("SELECT id, content FROM episodes WHERE user_id = ?", (user_id,)).fetchall()
        self.db.executemany(
            "INSERT INTO episodes_fts (episodes_fts, rowid, content) VALUES ('delete', ?, ?)",
            [(r["id"], r["content"]) for r in episodes],
        )
        self.db.execute("DELETE FROM facts_fts WHERE rowid IN (SELECT id FROM facts WHERE user_id = ?)", (user_id,))
        for table in ("pinned", "episodes", "facts", "summaries"):
            self.db.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> MemoryStore:  # noqa: PYI034 (typing.Self needs 3.11)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
