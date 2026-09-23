"""Compare chunking strategies on your own documents.

    python examples/compare_chunking.py                      # bundled sample
    python examples/compare_chunking.py --docs my_docs/ --questions my_q.jsonl

Documents are Markdown files. Questions are JSONL lines of
{"question": ..., "answer": ...}, where "answer" is a short string that
must appear in a retrieved passage for it to count as a hit.

Retrieval is BM25 only, so the comparison isolates chunking. Pass
--embed to add a sentence-transformers model and fuse with RRF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blueprints.chunking import STRATEGIES, parse_markdown
from blueprints.evaluate import Question, evaluate
from blueprints.retrieval import BM25, DenseIndex, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parent.parent


def load(docs_dir: Path, questions_path: Path):
    docs = [parse_markdown(p.stem, p.read_text()) for p in sorted(docs_dir.glob("*.md"))]
    questions = [Question(**json.loads(line)) for line in questions_path.read_text().splitlines() if line.strip()]
    return docs, questions


def build_retriever(chunks, embed=None):
    by_id = {c.id: c for c in chunks}
    bm25 = BM25().add_many((c.id, c.text) for c in chunks)
    dense = DenseIndex(embed).add_many((c.id, c.text) for c in chunks) if embed else None

    def retrieve(question: str, k: int) -> list[str]:
        rankings = [bm25.search(question, k=k * 4)]
        if dense:
            rankings.append(dense.search(question, k=k * 4))
        hits = reciprocal_rank_fusion(rankings, top=k * 4) if dense else rankings[0]
        # Parent-child: several children of one section collapse to one read.
        out, seen = [], set()
        for h in hits:
            text = by_id[h.id].read_text
            if text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) == k:
                break
        return out

    return retrieve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, default=ROOT / "data" / "handbook")
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=120, help="target chunk size in tokens")
    ap.add_argument("--embed", default=None, help="sentence-transformers model name, e.g. BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    docs, questions = load(args.docs, args.questions or args.docs / "questions.jsonl")
    embed = None
    if args.embed:
        from sentence_transformers import SentenceTransformer  # optional dependency

        model = SentenceTransformer(args.embed)
        embed = lambda texts: model.encode(list(texts), normalize_embeddings=True).tolist()

    print(f"{len(docs)} documents, {len(questions)} questions, k={args.k}, ~{args.tokens} tokens\n")
    for name, strategy in STRATEGIES.items():
        size_arg = "child_tokens" if name == "parent_child" else "max_tokens"
        chunks = [c for d in docs for c in strategy(d, **{size_arg: args.tokens})]
        report = evaluate(build_retriever(chunks, embed), questions, k=args.k)
        print(f"{name:<13} {len(chunks):>4} chunks   {report.row()}")
        for miss in report.misses:
            print(f"{'':<13} miss: {miss}")


if __name__ == "__main__":
    main()
