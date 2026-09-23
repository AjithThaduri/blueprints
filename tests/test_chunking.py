from pathlib import Path

from blueprints.chunking import STRATEGIES, count_tokens, parent_child, parse_markdown, recursive_split, structure_aware

MD = """# Guide

## Returns

### EU

The window is 14 days.

### US

The window is 30 days.
"""


def test_parse_keeps_heading_paths():
    doc = parse_markdown("g", MD)
    assert doc.title == "Guide"
    assert [s.path for s in doc.sections] == [["Guide", "Returns", "EU"], ["Guide", "Returns", "US"]]


def test_structure_aware_prefixes_section_path():
    chunks = structure_aware(parse_markdown("g", MD))
    assert chunks[0].text.startswith("Guide > Returns > EU")
    assert "14 days" in chunks[0].text


def test_parent_child_reads_whole_section():
    doc = parse_markdown("g", "# T\n\n## S\n\n" + " ".join(["word"] * 400) + " needle.")
    chunks = parent_child(doc, child_tokens=50)
    assert len(chunks) > 1
    assert all("needle" in c.read_text for c in chunks)


def test_recursive_split_respects_budget():
    text = "\n\n".join(" ".join(["alpha"] * 60) for _ in range(5))
    pieces = recursive_split(text, 100)
    assert all(count_tokens(p) <= 100 for p in pieces)
    assert sum(len(p.split()) for p in pieces) == 300


def test_every_strategy_runs_on_sample_corpus():
    root = Path(__file__).resolve().parent.parent / "data" / "handbook"
    for path in root.glob("*.md"):
        doc = parse_markdown(path.stem, path.read_text())
        for name, fn in STRATEGIES.items():
            kw = {"child_tokens": 80} if name == "parent_child" else {"max_tokens": 80}
            assert fn(doc, **kw), name
