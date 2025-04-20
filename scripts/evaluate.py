#!/usr/bin/env python
"""
Retrieval benchmark: dense (SBERT + FAISS) vs. a TF-IDF keyword baseline.

A query counts as a hit when a retrieved chunk contains the expected answer span.
Reports Hit@1, Hit@k, MRR and mean retrieval latency.

Usage:
    python scripts/evaluate.py [--docs data/sample] [--questions eval/questions.json] [--k 3]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunker import chunk_pages  # noqa: E402
from app.config import settings  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.parsers import parse  # noqa: E402
from app.store import VectorStore  # noqa: E402


def load_chunks(doc_dir: Path, size: int, overlap: int):
    chunks = []
    for path in sorted(doc_dir.iterdir()):
        for c in chunk_pages(parse(path.name, path.read_bytes()), size, overlap):
            chunks.append((path.name, c))
    return chunks


def score(rankings, questions):
    hit1 = hitk = rr = 0.0
    for ranked, q in zip(rankings, questions):
        ranks = [i for i, (fname, text) in enumerate(ranked) if fname == q["file"] and q["answer"].lower() in text.lower()]
        if ranks:
            hit1 += ranks[0] == 0
            hitk += 1
            rr += 1 / (ranks[0] + 1)
    n = len(questions)
    return {"hit@1": hit1 / n, "hit@k": hitk / n, "mrr": rr / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=60)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    chunks = load_chunks(Path(args.docs), args.chunk_size, args.overlap)
    texts = [c.text for _, c in chunks]
    print(f"{len(chunks)} chunks from {args.docs}, {len(questions)} questions, k={args.k}\n")

    # TF-IDF baseline
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words="english").fit(texts)
    matrix = tfidf.transform(texts)
    t0 = time.perf_counter()
    sparse_rank = []
    for q in questions:
        sims = (matrix @ tfidf.transform([q["question"]]).T).toarray().ravel()
        sparse_rank.append([(chunks[i][0], texts[i]) for i in np.argsort(-sims)[: args.k]])
    sparse_ms = (time.perf_counter() - t0) * 1000 / len(questions)

    # Dense retrieval through the same VectorStore the API uses
    embedder = Embedder(settings.embedding_model)
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp), embedder.dim)
        by_file: dict[str, list] = {}
        for fname, c in chunks:
            by_file.setdefault(fname, []).append(c)
        for fname, cs in by_file.items():
            store.add_document(fname, cs, embedder.encode([c.text for c in cs]))
        t0 = time.perf_counter()
        dense_rank = []
        for q in questions:
            hits = store.search(embedder.encode([q["question"]])[0], args.k)
            dense_rank.append([(h.chunk.filename, h.chunk.text) for h in hits])
        dense_ms = (time.perf_counter() - t0) * 1000 / len(questions)

    rows = [("TF-IDF baseline", score(sparse_rank, questions), sparse_ms), ("Dense (SBERT + FAISS)", score(dense_rank, questions), dense_ms)]
    print(f"| Retriever | Hit@1 | Hit@{args.k} | MRR | Latency / query |")
    print("|---|---|---|---|---|")
    for name, m, ms in rows:
        print(f"| {name} | {m['hit@1']:.2f} | {m['hit@k']:.2f} | {m['mrr']:.2f} | {ms:.1f} ms |")


if __name__ == "__main__":
    main()
