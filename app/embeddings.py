from functools import lru_cache

import numpy as np

from app.config import settings


class Embedder:
    def __init__(self, model_name: str = settings.embedding_model):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors, dtype="float32")


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
