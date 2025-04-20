"""FAISS vector store with JSON metadata, persisted to disk."""

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class ChunkRecord:
    id: int
    doc_id: str
    filename: str
    page: int
    ocr: bool
    text: str


@dataclass
class SearchHit:
    score: float
    chunk: ChunkRecord


class VectorStore:
    def __init__(self, directory: Path, dim: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.Lock()
        self._index_path = self.dir / "index.faiss"
        self._meta_path = self.dir / "chunks.json"
        self.chunks: dict[int, ChunkRecord] = {}
        self._next_id = 0
        self._load()

    def _load(self):
        if self._index_path.exists() and self._meta_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self.chunks = {c["id"]: ChunkRecord(**c) for c in raw}
            self._next_id = max(self.chunks, default=-1) + 1
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _save(self):
        faiss.write_index(self.index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps([asdict(c) for c in self.chunks.values()], ensure_ascii=False),
            encoding="utf-8",
        )

    def add_document(self, filename: str, chunks, vectors: np.ndarray) -> str:
        doc_id = uuid.uuid4().hex[:12]
        with self._lock:
            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype="int64")
            self._next_id += len(chunks)
            self.index.add_with_ids(vectors, ids)
            for cid, c in zip(ids.tolist(), chunks):
                self.chunks[cid] = ChunkRecord(cid, doc_id, filename, c.page, c.ocr, c.text)
            self._save()
        return doc_id

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
            if ids:
                self.index.remove_ids(np.array(ids, dtype="int64"))
                for cid in ids:
                    del self.chunks[cid]
                self._save()
        return len(ids)

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks.values():
            d = docs.setdefault(
                c.doc_id,
                {"doc_id": c.doc_id, "filename": c.filename, "chunks": 0, "pages": set(), "ocr_pages": set()},
            )
            d["chunks"] += 1
            d["pages"].add(c.page)
            if c.ocr:
                d["ocr_pages"].add(c.page)
        return [{**d, "pages": len(d["pages"]), "ocr_pages": len(d["ocr_pages"])} for d in docs.values()]

    def search(self, vector: np.ndarray, k: int, doc_ids: list[str] | None = None) -> list[SearchHit]:
        if self.index.ntotal == 0:
            return []
        # Over-fetch when filtering by document so the filter still yields k hits.
        fetch = min(self.index.ntotal, k * 4 if doc_ids else k)
        scores, ids = self.index.search(vector.reshape(1, -1), fetch)
        hits = []
        for score, cid in zip(scores[0], ids[0]):
            chunk = self.chunks.get(int(cid))
            if chunk is None or (doc_ids and chunk.doc_id not in doc_ids):
                continue
            hits.append(SearchHit(float(score), chunk))
            if len(hits) == k:
                break
        return hits
