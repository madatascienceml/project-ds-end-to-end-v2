"""Manual retrieval verification for the guidelines FAISS index (Block 14).

Loads the persisted index + chunk mapping from data/guidelines/index/,
runs 10 domain-relevant test queries, and prints the top-3 retrieved
chunks for each so retrieval quality can be checked by eye before an LLM
is layered on top. GATE: do not proceed to report generation until this
passes — if retrieval is broken, debugging the LLM instead wastes hours.

This is a standalone script, not a pytest test suite (despite the
`test_` filename, requested to match this file's purpose in the
workflow) — run it directly, not via `pytest`. All work happens inside
main(), guarded by `if __name__ == "__main__"`, so importing this module
(e.g. if pytest happens to collect it) has no side effects.

Reuses embed_texts() from indexer.py rather than loading its own
SentenceTransformer instance — the whole point of that shared function is
that index-build time and query time always use the same embedding model.
"""

import json
import sys
from pathlib import Path

import faiss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.indexer import CHUNKS_PATH, INDEX_PATH, embed_texts

TEST_QUERIES = [
    "What follow-up interval is recommended for a patient with no diabetic retinopathy?",
    "When should a patient be referred urgently to an ophthalmologist?",
    "What are the criteria for severe non-proliferative diabetic retinopathy?",
    "How should poor image quality or an ungradable image be handled?",
    "What is diabetic macular oedema and why can't a single fundus photo diagnose it?",
    "At what point after a diabetes diagnosis should retinopathy screening begin?",
    "What distinguishes mild from moderate non-proliferative retinopathy?",
    "What findings indicate proliferative diabetic retinopathy?",
    "Why is the follow-up interval for severe NPDR shorter than standard screening protocols?",
    "What acquisition factors, such as pupil dilation, affect fundus photography quality?",
]


def load_index(index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    """Load the persisted FAISS index and its chunk_id/text mapping."""
    index = faiss.read_index(str(index_path))
    with open(chunks_path, encoding="utf-8") as f:
        chunk_mapping = json.load(f)
    return index, chunk_mapping


def retrieve(query, index, chunk_mapping, top_k=3):
    """Embed a query with embed_texts() and return its top_k nearest chunks."""
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1:  # FAISS pads with -1 if top_k > number of indexed vectors
            continue
        entry = chunk_mapping[str(idx)]
        results.append({
            "rank": rank,
            "distance": float(dist),
            "chunk_id": entry["chunk_id"],
            "text": entry["text"],
        })
    return results


def _print_results(query, results):
    print(f"\nQuery: {query}")
    for r in results:
        preview = " ".join(r["text"].split())
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"  [{r['rank']}] {r['chunk_id']}  (L2 distance={r['distance']:.4f})")
        print(f"      {preview}")


def main():
    index, chunk_mapping = load_index()
    print(f"Loaded index: {index.ntotal} vectors, {len(chunk_mapping)} chunks")

    for query in TEST_QUERIES:
        results = retrieve(query, index, chunk_mapping, top_k=3)
        _print_results(query, results)


if __name__ == "__main__":
    main()
