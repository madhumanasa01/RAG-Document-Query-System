import hashlib
import re

import numpy as np
import pytest


class HashEmbedder:
    """Deterministic bag-of-words embedder so tests run without downloading a model."""

    dim = 256

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def client(tmp_path, embedder, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main
    from app.config import settings
    from app.store import VectorStore

    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "min_score", 0.0)
    main.state.clear()
    main.state["embedder"] = embedder
    main.state["store"] = VectorStore(tmp_path, embedder.dim)
    with TestClient(main.app) as c:
        yield c
    main.state.clear()
