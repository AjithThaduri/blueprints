"""Retrieval metrics. Measure retrieval on its own before judging answers:
if the right passage never comes back, no prompt can fix it."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class Question:
    question: str
    answer: str  # a short string that must appear in a passage for it to count


@dataclass
class Report:
    recall_at_k: float
    mrr: float
    n: int
    k: int
    misses: list[str]

    def row(self) -> str:
        return f"recall@{self.k}={self.recall_at_k:.2f}  mrr={self.mrr:.2f}  (n={self.n})"


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def evaluate(
    retrieve: Callable[[str, int], Sequence[str]],
    questions: Sequence[Question],
    k: int = 3,
) -> Report:
    """``retrieve(question, k)`` returns the texts a model would read, in rank
    order. A hit is the first passage containing the expected answer."""
    hits, rr, misses = 0, 0.0, []
    for q in questions:
        passages = retrieve(q.question, k)
        rank = next((i for i, p in enumerate(passages, 1) if _norm(q.answer) in _norm(p)), None)
        if rank is None:
            misses.append(q.question)
        else:
            hits += 1
            rr += 1.0 / rank
    n = len(questions) or 1
    return Report(hits / n, rr / n, len(questions), k, misses)
