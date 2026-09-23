"""Assemble one turn's context from memory tiers, inside a token budget.

The agent never sees the whole history. Each turn gets: pinned facts,
the rolling summary, the recent window verbatim, and whatever retrieval
finds relevant to this message — each tier capped separately so one tier
can't crowd out the others.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..chunking import count_tokens
from .store import MemoryStore


@dataclass
class Budget:
    pinned: int = 800
    summary: int = 1200
    window: int = 6000
    retrieved: int = 4000
    window_turns: int = 8


@dataclass
class TurnContext:
    pinned: list[str] = field(default_factory=list)
    summary: str = ""
    recent: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = []
        if self.pinned:
            parts.append("## About this user\n" + "\n".join(f"- {p}" for p in self.pinned))
        if self.facts:
            parts.append("## Relevant facts\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.episodes:
            parts.append("## Relevant earlier messages\n" + "\n".join(self.episodes))
        if self.summary:
            parts.append("## Earlier in this conversation\n" + self.summary)
        if self.recent:
            parts.append("## Recent turns\n" + "\n".join(self.recent))
        return "\n\n".join(parts)

    @property
    def tokens(self) -> int:
        return count_tokens(self.render()) if self.render() else 0


def _fit(items: list[str], budget: int, from_end: bool = False) -> list[str]:
    """Keep whole items until the budget runs out. from_end keeps the newest."""
    seq = list(reversed(items)) if from_end else items
    out, used = [], 0
    for item in seq:
        cost = count_tokens(item)
        if used + cost > budget:
            break
        out.append(item)
        used += cost
    return list(reversed(out)) if from_end else out


def build_context(store: MemoryStore, user_id: str, message: str, budget: Budget | None = None) -> TurnContext:
    b = budget or Budget()
    recent = store.recent(user_id, b.window_turns)
    summary, _ = store.summary(user_id)

    facts = [f.text for f in store.search_facts(user_id, message, k=20)]
    earlier = [
        f"[{e.role}] {e.content}" for e in store.search_episodes(user_id, message, k=10, exclude_recent=b.window_turns)
    ]
    # Facts first: they're denser than raw episodes for the same tokens.
    retrieved_facts = _fit(facts, b.retrieved)
    left = b.retrieved - sum(count_tokens(f) for f in retrieved_facts)

    return TurnContext(
        pinned=_fit([f"{k}: {v}" for k, v in store.pinned(user_id).items()], b.pinned),
        summary=summary if count_tokens(summary) <= b.summary else " ".join(summary.split()[: int(b.summary * 0.75)]),
        recent=_fit([f"[{e.role}] {e.content}" for e in recent], b.window, from_end=True),
        facts=retrieved_facts,
        episodes=_fit(earlier, max(left, 0)),
    )


Extractor = Callable[[str, list[str]], list[tuple[str, str, str]]]
Summariser = Callable[[str, list[str]], str]


def consolidate(
    store: MemoryStore,
    user_id: str,
    extract: Extractor,
    summarise: Summariser,
    window_turns: int = 8,
) -> None:
    """The write path. Run it after the reply, off the hot path (a queue,
    a background task). ``extract`` returns (subject, predicate, value)
    triples from the summary plus new turns; ``summarise`` folds turns that
    have left the recent window into the rolling summary.

    Both are plain callables so you can back them with any model. Use a
    capable model for extraction: small models without structured output
    are where memory pipelines usually break.
    """
    summary, through = store.summary(user_id)
    all_recent = store.recent(user_id, window_turns * 2)
    new_turns = [e for e in all_recent if e.id > through]
    if not new_turns:
        return

    for subject, predicate, value in extract(summary, [f"[{e.role}] {e.content}" for e in new_turns]):
        store.add_fact(user_id, subject, predicate, value, source_episode=new_turns[-1].id)

    window_ids = {e.id for e in store.recent(user_id, window_turns)}
    leaving = [e for e in new_turns if e.id not in window_ids]
    if leaving:
        store.set_summary(user_id, summarise(summary, [f"[{e.role}] {e.content}" for e in leaving]), leaving[-1].id)
