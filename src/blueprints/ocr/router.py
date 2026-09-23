"""Tiered OCR routing, from "Open-source OCR: route pages, don't pick a model".

Each page goes to the cheapest reader that can handle it:

  tier 0  the PDF's own text layer, if it exists and is trustworthy
  tier 1  a small document model (or a classic engine on CPU)
  tier 2  a larger model, only for pages tier 1 flags as unreliable

Readers are plain callables so you can plug in anything: Tesseract,
PaddleOCR, or a vision-language model behind an OpenAI-compatible server
such as vLLM (see ``openai_compatible_reader``).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_VOWELS = set("aeiouyAEIOUY")
# Characters that show up when a text layer is mis-encoded or glyph-mapped.
_JUNK = set("�□■•·\u0000") | {chr(c) for c in range(0xE000, 0xE100)}


@dataclass
class Quality:
    score: float
    word_like: float  # share of tokens that look like words
    junk: float  # share of junk / replacement characters
    coverage: float  # characters per page area, normalised to 0..1
    reasons: list[str] = field(default_factory=list)


def text_layer_quality(text: str, page_area: float = 1.0, expected_chars: float = 1800.0) -> Quality:
    """Score an embedded text layer without any model.

    - word_like: tokens that contain a vowel and aren't absurdly long. Garbled
      layers ("Tlie qnick brovvn") still pass this, so it's paired with junk.
    - junk: replacement and private-use glyphs from broken font mappings.
    - coverage: a scanned page with a thin, invisible OCR layer has far fewer
      characters than a born-digital page of the same size.

    ``expected_chars`` is roughly a full page of body text; tune it per corpus.
    """
    reasons = []
    stripped = text.strip()
    if not stripped:
        return Quality(0.0, 0.0, 0.0, 0.0, ["empty text layer"])

    tokens = _WORD.findall(stripped)
    word_like = (
        sum(1 for t in tokens if 1 < len(t) <= 20 and (set(t) & _VOWELS or not t.isascii())) / len(tokens)
        if tokens
        else 0.0
    )
    counts = Counter(stripped)
    junk = sum(n for ch, n in counts.items() if ch in _JUNK) / len(stripped)
    coverage = min(len(stripped) / (expected_chars * page_area), 1.0)

    if word_like < 0.7:
        reasons.append(f"only {word_like:.0%} of tokens look like words")
    if junk > 0.01:
        reasons.append(f"{junk:.1%} junk characters")
    if coverage < 0.15:
        reasons.append("very little text for the page size")

    score = 0.6 * word_like + 0.25 * coverage + 0.15 * (1 - min(junk * 20, 1.0))
    return Quality(round(score, 3), round(word_like, 3), round(junk, 4), round(coverage, 3), reasons)


class Page(Protocol):
    number: int
    area: float  # page area relative to a letter/A4 page (1.0 = full page)

    def text_layer(self) -> str: ...
    def image(self) -> bytes: ...


@dataclass
class ReadResult:
    text: str
    confidence: float  # 0..1, as reported or estimated by the reader
    flags: list[str] = field(default_factory=list)  # e.g. "handwriting", "table"


Reader = Callable[[bytes], ReadResult]


@dataclass
class PageOutput:
    number: int
    text: str
    tier: int
    reason: str


@dataclass
class Router:
    small: Reader
    large: Reader | None = None
    trust_text_layer_above: float = 0.85
    escalate_below: float = 0.8
    escalate_flags: tuple[str, ...] = ("handwriting", "form", "degraded")
    stats: Counter = field(default_factory=Counter)

    def read(self, page: Page) -> PageOutput:
        q = text_layer_quality(page.text_layer(), page.area)
        if q.score >= self.trust_text_layer_above and not q.reasons:
            self.stats[0] += 1
            return PageOutput(page.number, page.text_layer(), 0, "text layer trusted")

        image = page.image()
        first = self.small(image)
        needs_more = first.confidence < self.escalate_below or any(f in self.escalate_flags for f in first.flags)
        if needs_more and self.large is not None:
            second = self.large(image)
            self.stats[2] += 1
            why = f"confidence {first.confidence:.2f}" + (f", flags {first.flags}" if first.flags else "")
            return PageOutput(page.number, second.text, 2, f"escalated: {why}")

        self.stats[1] += 1
        reason = "; ".join(q.reasons) or "text layer below threshold"
        return PageOutput(page.number, first.text, 1, f"model read ({reason})")

    def report(self) -> str:
        total = sum(self.stats.values()) or 1
        return "  ".join(f"tier {t}: {self.stats[t]} ({self.stats[t] / total:.0%})" for t in (0, 1, 2))


def openai_compatible_reader(base_url: str, model: str, prompt: str | None = None, timeout: float = 120.0) -> Reader:
    """A Reader backed by any OpenAI-compatible vision endpoint — for example
    ``vllm serve allenai/olmOCR-2-7B-1025-FP8`` or a PaddleOCR-VL server.

    Confidence isn't exposed by most servers, so it's estimated from the
    output's own text quality. Requires ``httpx`` (``pip install httpx``).
    """
    import base64

    import httpx

    instruction = prompt or (
        "Transcribe this page to Markdown. Keep headings, lists and tables. "
        "Write [illegible] for anything you cannot read. Output only the transcription."
    )

    def read(image: bytes) -> ReadResult:
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()},
                        },
                    ],
                }
            ],
        }
        r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        illegible = text.count("[illegible]")
        confidence = text_layer_quality(text).word_like * (0.5 if illegible > 3 else 1.0)
        return ReadResult(text, confidence, ["degraded"] if illegible > 3 else [])

    return read
