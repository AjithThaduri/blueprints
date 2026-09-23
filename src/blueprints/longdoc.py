"""Long-document retrieval with a map, from "RAG for long documents".

Build a tree from the document's own headings, summarise upwards, index
passages and summaries together, route each question, and hand the model
an excerpt in reading order with a breadcrumb on every piece.

Summaries are pluggable. The default is extractive (leading sentences),
which cannot state anything the source doesn't — swap in a model-written
summariser once you have a faithfulness check in place.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .chunking import Document, Section
from .retrieval import BM25

Route = Literal["local", "cross_ref", "global"]
Summariser = Callable[[str, list[str]], str]


@dataclass
class Node:
    id: str
    path: list[str]
    text: str  # passage text, or summary for inner nodes
    position: int  # reading order
    is_summary: bool = False
    children: list[Node] = field(default_factory=list)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)


def extractive_summary(title: str, texts: list[str], sentences: int = 2) -> str:
    """Lead sentences of each child. Faithful by construction."""
    out = []
    for t in texts:
        parts = re.split(r"(?<=[.!?])\s+", t.strip())
        out.append(" ".join(parts[:sentences]))
    return " ".join(out)  # the breadcrumb is stored separately, not repeated here


@dataclass
class DocumentMap:
    doc: Document
    nodes: dict[str, Node]
    root: Node
    index: BM25

    @classmethod
    def build(cls, doc: Document, summarise: Summariser | None = None) -> DocumentMap:
        summarise = summarise or extractive_summary
        root = Node(f"{doc.id}:root", [doc.title], "", -1, is_summary=True)
        nodes: dict[str, Node] = {root.id: root}

        def ensure(path: list[str]) -> Node:
            parent = root
            for depth in range(1, len(path) + 1):
                key = f"{doc.id}:" + "/".join(path[:depth])
                if key not in nodes:
                    node = Node(key, [doc.title, *path[:depth]], "", 10_000, is_summary=True)
                    nodes[key] = node
                    parent.children.append(node)
                parent = nodes[key]
            return parent

        for s in doc.sections:
            path = [p for p in s.path if p != doc.title]
            parent = ensure(path) if path else root
            leaf = Node(f"{doc.id}:s{s.order}", [doc.title, *path], s.text, s.order)
            nodes[leaf.id] = leaf
            parent.children.append(leaf)

        def fill(node: Node) -> int:
            if not node.is_summary:
                return node.position
            positions = [fill(c) for c in node.children]
            node.position = min(positions) if positions else node.position
            child_texts = [c.text for c in node.children if c.text]
            node.text = summarise(node.breadcrumb, child_texts) if child_texts else ""
            return node.position

        fill(root)
        index = BM25().add_many((n.id, f"{n.breadcrumb}\n{n.text}") for n in nodes.values() if n.text)
        return cls(doc, nodes, root, index)

    # ---------------------------------------------------------- routing
    GLOBAL_CUES = re.compile(
        r"\b(overall|summar\w*|main (points|themes|risks)|in general|whole|entire|key takeaways|what does .* cover)\b",
        re.IGNORECASE,
    )
    XREF_CUES = re.compile(
        r"\b(section|schedule|appendix|clause|compare|difference|versus|vs\.?|both)\b", re.IGNORECASE
    )

    def route(self, question: str) -> Route:
        """Cheap rules first. Log every routed question: the misroutes you
        find there are the training data for a small classifier later."""
        if self.GLOBAL_CUES.search(question):
            return "global"
        if self.XREF_CUES.search(question):
            return "cross_ref"
        return "local"

    # -------------------------------------------------------- retrieval
    def retrieve(self, question: str, k: int = 6) -> tuple[Route, list[Node]]:
        kind = self.route(question)
        hits = [self.nodes[h.id] for h in self.index.search(question, k=k * 3)]
        if kind == "global":
            picked = [self.root] + [n for n in hits if n.is_summary]
        elif kind == "cross_ref":
            picked = []
            for n in hits:
                picked.extend(self._leaves(n) if n.is_summary else [n])
        else:
            picked = [n for n in hits if not n.is_summary] or hits

        seen, out = set(), []
        for n in picked:
            if n.id not in seen and n.text:
                seen.add(n.id)
                out.append(n)
            if len(out) == k:
                break
        return kind, sorted(out, key=lambda n: n.position)  # reading order, not score order

    def _leaves(self, node: Node) -> list[Node]:
        if not node.children:
            return [node]
        return [leaf for c in node.children for leaf in self._leaves(c)]

    def excerpt(self, question: str, k: int = 6) -> str:
        """The context a model would read, with breadcrumbs for citation."""
        _, nodes = self.retrieve(question, k)
        return "\n\n".join(f"[{n.breadcrumb}]\n{n.text}" for n in nodes)


__all__ = ["DocumentMap", "Node", "Section", "extractive_summary"]
