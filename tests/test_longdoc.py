from pathlib import Path

from blueprints.chunking import parse_markdown
from blueprints.longdoc import DocumentMap

ROOT = Path(__file__).resolve().parent.parent / "data" / "handbook"


def build():
    return DocumentMap.build(parse_markdown("policies", (ROOT / "policies.md").read_text()))


def test_routes_by_question_shape():
    m = build()
    assert m.route("Summarise the main points of this handbook") == "global"
    assert m.route("Compare the returns section for the EU and the UK") == "cross_ref"
    assert m.route("How long is the battery warranty?") == "local"


def test_local_question_finds_the_passage():
    kind, nodes = build().retrieve("How long is the warranty on batteries?", k=3)
    assert kind == "local"
    assert any("two years" in n.text for n in nodes)


def test_global_question_starts_from_summaries():
    kind, nodes = build().retrieve("Give me an overall summary of the policies", k=4)
    assert kind == "global"
    assert nodes and all(n.is_summary for n in nodes)


def test_excerpt_is_in_reading_order_with_breadcrumbs():
    m = build()
    _, nodes = m.retrieve("Compare returns in the EU versus the United States", k=6)
    positions = [n.position for n in nodes]
    assert positions == sorted(positions)
    assert m.excerpt("Compare returns in the EU versus the United States").startswith("[Harbor Tools")


def test_extractive_summaries_only_contain_source_sentences():
    m = build()
    source = (ROOT / "policies.md").read_text()
    for node in m.nodes.values():
        if node.is_summary and node.text:
            for sentence in filter(None, (s.strip() for s in node.text.split(". "))):
                assert sentence.rstrip(".") in source
