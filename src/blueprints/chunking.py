"""Chunking strategies from the "Chunking strategies, compared" blueprint.

Every strategy takes a parsed Document and returns Chunks. A Chunk carries
the text that gets indexed and, separately, the text a model should read
(its parent), so parent-child retrieval needs no special casing downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def count_tokens(text: str) -> int:
    """Rough token count (~0.75 words per token). Swap in a real tokenizer
    for production budgets; for comparing strategies this is enough."""
    return max(1, round(len(text.split()) / 0.75))


@dataclass
class Section:
    path: list[str]
    text: str
    order: int


@dataclass
class Document:
    id: str
    title: str
    sections: list[Section]

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str  # what gets indexed
    read_text: str = ""  # what the model reads; defaults to text
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.read_text:
            self.read_text = self.text


def parse_markdown(doc_id: str, markdown: str) -> Document:
    """Split Markdown into sections along its headings, keeping the path."""
    title = doc_id
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section([h for _, h in stack], body, len(sections)))
        buf.clear()

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2)
            if level == 1 and title == doc_id:
                title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()
    return Document(doc_id, title, sections)


# --------------------------------------------------------------- splitting


def fixed_size(doc: Document, max_tokens: int = 200, overlap: float = 0.15) -> list[Chunk]:
    """Cut every N tokens with overlap, ignoring structure. The baseline."""
    words = doc.text.split()
    size = max(1, round(max_tokens * 0.75))
    step = max(1, round(size * (1 - overlap)))
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = " ".join(words[start : start + size])
        chunks.append(Chunk(f"{doc.id}#f{i}", doc.id, piece))
        if start + size >= len(words):
            break
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_split(text: str, max_tokens: int, separators: list[str] | None = None) -> list[str]:
    """Split on the largest separator that yields pieces under max_tokens,
    falling back to smaller separators only for pieces still too long."""
    separators = _SEPARATORS if separators is None else separators
    if count_tokens(text) <= max_tokens or not separators:
        return [text.strip()] if text.strip() else []
    sep, rest = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return recursive_split(text, max_tokens, rest)

    out: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            out.append(current.strip())
        if count_tokens(part) > max_tokens:
            out.extend(recursive_split(part, max_tokens, rest))
            current = ""
        else:
            current = part
    if current.strip():
        out.append(current.strip())
    return out


def recursive(doc: Document, max_tokens: int = 200) -> list[Chunk]:
    """Separator-aware splitting of the whole text, no section awareness."""
    return [Chunk(f"{doc.id}#r{i}", doc.id, piece) for i, piece in enumerate(recursive_split(doc.text, max_tokens))]


def _header(doc: Document, section: Section) -> str:
    return " > ".join([doc.title, *[p for p in section.path if p != doc.title]])


def structure_aware(doc: Document, max_tokens: int = 200, with_header: bool = True) -> list[Chunk]:
    """Split along sections; recurse inside long ones; prefix the section path.

    The header line is the cheapest improvement in the whole blueprint: a
    chunk that says which document and section it came from matches
    questions that name the topic but not the chunk's exact words.
    """
    chunks = []
    for s in doc.sections:
        header = _header(doc, s)
        for j, piece in enumerate(recursive_split(s.text, max_tokens)):
            text = f"{header}\n\n{piece}" if with_header else piece
            chunks.append(Chunk(f"{doc.id}#s{s.order}.{j}", doc.id, text, meta={"section": header, "order": s.order}))
    return chunks


def parent_child(doc: Document, child_tokens: int = 100, with_header: bool = True) -> list[Chunk]:
    """Index small pieces for precise matching; read the whole section."""
    chunks = []
    for s in doc.sections:
        header = _header(doc, s)
        parent = f"{header}\n\n{s.text}"
        for j, piece in enumerate(recursive_split(s.text, child_tokens)):
            text = f"{header}\n\n{piece}" if with_header else piece
            chunks.append(
                Chunk(
                    f"{doc.id}#p{s.order}.{j}",
                    doc.id,
                    text,
                    read_text=parent,
                    meta={"section": header, "order": s.order, "parent": f"{doc.id}#p{s.order}"},
                )
            )
    return chunks


STRATEGIES = {
    "fixed": fixed_size,
    "recursive": recursive,
    "structure": structure_aware,
    "parent_child": parent_child,
}
