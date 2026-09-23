"""Adapter from a real PDF to the Router's Page protocol, using pypdf for the
text layer and pypdfium2 to render images (both permissively licensed).

    pip install "blueprints[pdf]"
"""

from __future__ import annotations

import io
from dataclasses import dataclass

LETTER_AREA = 612 * 792  # points


@dataclass
class PdfPage:
    number: int
    area: float
    _text: str
    _render: object

    def text_layer(self) -> str:
        return self._text

    def image(self) -> bytes:
        return self._render()  # type: ignore[operator]


def open_pdf(path: str, longest_side: int = 1288):
    """Yield pages. ``longest_side`` defaults to what olmOCR was trained on;
    match whatever model sits behind your readers."""
    import pypdf
    import pypdfium2

    reader = pypdf.PdfReader(path)
    doc = pypdfium2.PdfDocument(path)
    for i, page in enumerate(reader.pages):
        w, h = float(page.mediabox.width), float(page.mediabox.height)

        def render(i=i, w=w, h=h) -> bytes:
            scale = longest_side / max(w, h)
            img = doc[i].render(scale=scale).to_pil()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        yield PdfPage(i + 1, (w * h) / LETTER_AREA, page.extract_text() or "", render)
