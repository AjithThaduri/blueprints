"""Small, dependency-free retrieval primitives.

BM25 for keyword search, an optional dense index that takes any embedding
function, and reciprocal rank fusion to combine them. Enough to compare
chunking strategies honestly without standing up a vector database.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# A short stoplist is enough for BM25; the IDF term handles the rest.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    id: str
    score: float


@dataclass
class BM25:
    """Okapi BM25 over an in-memory corpus.

    k1 controls term-frequency saturation, b controls length normalisation.
    The defaults (1.2, 0.75) are the usual starting point.
    """

    k1: float = 1.2
    b: float = 0.75
    _docs: dict[str, Counter] = field(default_factory=dict, repr=False)
    _lengths: dict[str, int] = field(default_factory=dict, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            raise ValueError(f"duplicate id: {doc_id}")
        terms = Counter(tokenize(text))
        self._docs[doc_id] = terms
        self._lengths[doc_id] = sum(terms.values())
        self._df.update(terms.keys())

    def add_many(self, items: Iterable[tuple[str, str]]) -> BM25:
        for doc_id, text in items:
            self.add(doc_id, text)
        return self

    def search(self, query: str, k: int = 10) -> list[Hit]:
        n = len(self._docs)
        if n == 0:
            return []
        avg_len = sum(self._lengths.values()) / n
        q_terms = tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        for term in set(q_terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, terms in self._docs.items():
                tf = terms.get(term, 0)
                if tf == 0:
                    continue
                norm = 1 - self.b + self.b * self._lengths[doc_id] / avg_len
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [Hit(doc_id, score) for doc_id, score in ranked[:k]]


EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass
class DenseIndex:
    """Cosine-similarity search over vectors from any embedding function.

    Pass e.g. a sentence-transformers model's ``encode``; nothing here
    depends on a particular provider.
    """

    embed: EmbedFn
    _ids: list[str] = field(default_factory=list, repr=False)
    _vecs: list[list[float]] = field(default_factory=list, repr=False)

    def add_many(self, items: Iterable[tuple[str, str]]) -> DenseIndex:
        items = list(items)
        vectors = self.embed([text for _, text in items])
        for (doc_id, _), vec in zip(items, vectors):
            self._ids.append(doc_id)
            self._vecs.append(_normalise(vec))
        return self

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = _normalise(self.embed([query])[0])
        scored = [(doc_id, sum(a * b for a, b in zip(q, v))) for doc_id, v in zip(self._ids, self._vecs)]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [Hit(doc_id, score) for doc_id, score in scored[:k]]


def _normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = 60, top: int = 10) -> list[Hit]:
    """Merge ranked lists by summing 1 / (k + rank).

    Rank-based, so it needs no score calibration between BM25 and cosine.
    k=60 is the constant from the original RRF paper (Cormack et al., 2009).
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.id] += 1.0 / (k + rank)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Hit(doc_id, score) for doc_id, score in ranked[:top]]
