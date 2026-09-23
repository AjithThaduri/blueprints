# blueprints

Runnable reference code for the architecture write-ups at
**[ajiththaduri.site/blueprints](https://www.ajiththaduri.site/blueprints)**.

Each article explains a design and its trade-offs; this repo is the working
version you can run, test and adapt. The core has **no dependencies** — the
Python standard library, a small BM25 implementation and SQLite's built-in
full-text search — so you can clone it and have results in a minute. Model
backends (embeddings, OCR models, LLMs) plug in as ordinary callables.

```bash
git clone https://github.com/AjithThaduri/blueprints && cd blueprints
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # all tests, no network, no GPU
python examples/compare_chunking.py      # chunking strategies on a sample corpus
```

| Blueprint | Module | What you can run |
|---|---|---|
| [Chunking strategies, compared](https://www.ajiththaduri.site/blueprints/chunking-strategies-compared) | `blueprints.chunking`, `blueprints.evaluate` | Compare fixed, recursive, structure-aware and parent–child chunking on **your** documents and see Recall@k and MRR |
| [RAG for long documents](https://www.ajiththaduri.site/blueprints/long-document-rag) | `blueprints.longdoc` | Build a heading tree with summaries, route questions local / cross-reference / global, get an excerpt in reading order |
| [Open-source OCR: route pages](https://www.ajiththaduri.site/blueprints/open-source-ocr-pipeline) | `blueprints.ocr` | Score PDF text layers without a model, and route pages across tiers of OCR readers (any vLLM-served VLM plugs in) |
| [Agent memory without replaying history](https://www.ajiththaduri.site/blueprints/memory-aware-agents) | `blueprints.memory` | Tiered memory on SQLite: facts with validity windows, per-user isolation, real deletion, context assembly on a token budget |
| [RAG patterns: which ones are worth it](https://www.ajiththaduri.site/blueprints/rag-patterns-field-guide) | `blueprints.retrieval` | BM25, pluggable dense search, and reciprocal rank fusion |

---

## Chunking: measure it on your own documents

```bash
python examples/compare_chunking.py --docs path/to/markdown/ --questions questions.jsonl --k 3 --tokens 150
```

`questions.jsonl` holds lines like `{"question": "...", "answer": "..."}`,
where `answer` is a short string that must appear in a retrieved passage for
it to count. Fifty real questions is enough to see differences.

Retrieval is BM25-only by default so the comparison isolates chunking. Add
`--embed BAAI/bge-small-en-v1.5` (with `pip install -e ".[embed]"`) to fuse in
dense retrieval with RRF.

The bundled sample (`data/handbook/`) is deliberately awkward: three returns
sections that differ only by region heading, a shipping table, and sentences
that depend on the one before. Chunks without their section path can't tell
the regions apart. On it, with BM25 and ~120-token chunks:

```
fixed            9 chunks   recall@3=0.83  mrr=0.70  (n=18)
recursive        9 chunks   recall@3=0.78  mrr=0.69  (n=18)
structure       12 chunks   recall@3=0.94  mrr=0.92  (n=18)
parent_child    12 chunks   recall@3=0.94  mrr=0.92  (n=18)
```

Eighteen questions on one small corpus proves nothing general — it's there
so you can see the mechanics before pointing it at your own documents. The
one question every strategy misses ("How much does Harbor Pro cost in the
EU?") is instructive too: the text says "costs" and "European Union", and
keyword search can't bridge that. That's the gap dense retrieval fills.

## Long documents

```python
from blueprints.chunking import parse_markdown
from blueprints.longdoc import DocumentMap

doc_map = DocumentMap.build(parse_markdown("policies", open("data/handbook/policies.md").read()))
doc_map.route("Summarise the main points")  # "global"
print(doc_map.excerpt("Compare returns in the EU and the UK"))
```

Summaries default to **extractive** (leading sentences), which can't say
anything the source doesn't. Pass `summarise=` to use a model once you have a
faithfulness check; the test suite includes one you can adapt.

Routing starts as rules on purpose. Log routed questions: the misroutes are
your training set for a small classifier later.

## OCR routing

```python
from blueprints.ocr import Router, openai_compatible_reader
from blueprints.ocr.pdf import open_pdf  # pip install -e ".[pdf,vlm]"

router = Router(
    small=openai_compatible_reader("http://localhost:8000/v1", "PaddlePaddle/PaddleOCR-VL"),
    large=openai_compatible_reader("http://localhost:8001/v1", "allenai/olmOCR-2-7B-1025-FP8"),
)
for page in open_pdf("scan.pdf"):
    out = router.read(page)
    print(out.number, out.tier, out.reason)
print(router.report())  # share of pages per tier
```

`text_layer_quality` decides whether a page's embedded text can be trusted,
using word shape, junk-glyph ratio and text coverage — no model needed.
Model names above are examples; check each model's **weights licence**, which
can differ from its code licence.

## Agent memory

```python
from blueprints.memory import MemoryStore, build_context, consolidate

mem = MemoryStore("memory.db")
mem.pin("u1", "role", "platform engineer")
mem.log("u1", "user", "The launch moved from Friday to Monday.")
mem.add_fact("u1", "launch", "date", "Friday")
mem.add_fact("u1", "launch", "date", "Monday")  # closes the Friday fact, keeps history

ctx = build_context(mem, "u1", "When is the launch?")
print(ctx.render())
```

- Changed facts are **invalidated, not overwritten**, so "what did we think
  last week?" stays answerable.
- Every read and write is scoped by `user_id`; `forget_user` removes a user
  from every tier and search index.
- `consolidate(store, user, extract, summarise)` is the write path. Run it
  after the reply, off the request path, with whatever model you trust for
  extraction.
- The schema maps onto Postgres (`tsvector` + `pgvector`) when SQLite is no
  longer enough.

---

## Design choices

- **No framework lock-in.** Readers, embedders, summarisers and extractors
  are plain callables.
- **Tests over claims.** Every behaviour the articles describe has a test.
  Run `pytest` before trusting any of it.
- **Small on purpose.** This is reference code to read and adapt, not a
  framework. For production retrieval engines, see the "Repos to explore"
  section at the end of each article.

## Credits

The ideas here build on published work, credited in each article: Okapi BM25;
reciprocal rank fusion (Cormack et al., 2009); Anthropic's contextual
retrieval; RAPTOR (Sarthi et al., 2024); olmOCR and OmniDocBench; MemGPT/Letta,
Mem0 and Zep/Graphiti (bi-temporal facts). The code is original.

## Licence

MIT — see [LICENSE](LICENSE). The sample handbook in `data/` is fictional.
