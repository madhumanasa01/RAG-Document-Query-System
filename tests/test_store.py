from app.chunker import Chunk
from app.store import VectorStore


def _add(store, embedder, filename, texts):
    chunks = [Chunk(t, 1, False) for t in texts]
    return store.add_document(filename, chunks, embedder.encode(texts))


def test_search_returns_most_similar_chunk(tmp_path, embedder):
    store = VectorStore(tmp_path, embedder.dim)
    _add(store, embedder, "a.txt", ["the warranty lasts two years", "shipping takes five days"])
    hits = store.search(embedder.encode(["how long is the warranty"])[0], k=1)
    assert hits[0].chunk.text == "the warranty lasts two years"


def test_persistence_round_trip(tmp_path, embedder):
    store = VectorStore(tmp_path, embedder.dim)
    doc_id = _add(store, embedder, "a.txt", ["alpha beta", "gamma delta"])
    reloaded = VectorStore(tmp_path, embedder.dim)
    assert reloaded.index.ntotal == 2
    assert reloaded.documents()[0]["doc_id"] == doc_id


def test_delete_document(tmp_path, embedder):
    store = VectorStore(tmp_path, embedder.dim)
    a = _add(store, embedder, "a.txt", ["alpha beta"])
    _add(store, embedder, "b.txt", ["gamma delta"])
    assert store.delete_document(a) == 1
    assert store.index.ntotal == 1
    assert [d["filename"] for d in store.documents()] == ["b.txt"]


def test_filter_by_document(tmp_path, embedder):
    store = VectorStore(tmp_path, embedder.dim)
    _add(store, embedder, "a.txt", ["refund policy is thirty days"])
    b = _add(store, embedder, "b.txt", ["refund requests need a receipt"])
    hits = store.search(embedder.encode(["refund policy"])[0], k=5, doc_ids=[b])
    assert {h.chunk.filename for h in hits} == {"b.txt"}
