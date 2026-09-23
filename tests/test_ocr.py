from dataclasses import dataclass

from blueprints.ocr import ReadResult, Router, text_layer_quality

GOOD = (
    "Customers can return unused items for a full refund. The return window is thirty days from delivery. "
    "Items must be in their original packaging, and the customer pays return shipping. "
) * 8


def test_good_text_layer_scores_high():
    q = text_layer_quality(GOOD)
    assert q.score > 0.85 and not q.reasons


def test_empty_and_garbled_layers_score_low():
    assert text_layer_quality("").score == 0
    garbled = "�� Tl�e q�ck brvvn fx jmps " * 10
    q = text_layer_quality(garbled)
    assert q.score < 0.85 and q.reasons


def test_thin_layer_on_a_full_page_is_suspicious():
    q = text_layer_quality("Page 3", page_area=1.0)
    assert "very little text for the page size" in q.reasons


@dataclass
class FakePage:
    number: int
    text: str
    area: float = 1.0

    def text_layer(self):
        return self.text

    def image(self):
        return b"png"


def test_router_uses_cheapest_adequate_tier():
    calls = []

    def small(img):
        calls.append("small")
        return ReadResult("scanned text", 0.95)

    def large(img):
        calls.append("large")
        return ReadResult("carefully read text", 0.99)

    r = Router(small=small, large=large)
    assert r.read(FakePage(1, GOOD)).tier == 0
    assert r.read(FakePage(2, "")).tier == 1
    assert calls == ["small"]


def test_router_escalates_low_confidence_and_flagged_pages():
    r = Router(
        small=lambda img: ReadResult("???", 0.4),
        large=lambda img: ReadResult("clean", 0.99),
    )
    out = r.read(FakePage(1, ""))
    assert out.tier == 2 and out.text == "clean" and "confidence" in out.reason

    r2 = Router(small=lambda img: ReadResult("note", 0.95, ["handwriting"]), large=lambda img: ReadResult("n", 0.9))
    assert r2.read(FakePage(1, "")).tier == 2
    assert "tier 2: 1 (100%)" in r2.report()
