"""Embedding providers behind a common interface.

Default is a local sentence-transformers model (offline, no API key). Set
``embedding_provider="gemini"`` in the config to use Gemini instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from config import Config


class BaseEmbedder(ABC):
    """Common interface for all embedding backends."""

    dim: int

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of documents -> (n, dim) float32 array."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query -> (dim,) float32 array."""


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers backend. Downloads the model once, then offline."""

    def __init__(self, model_name: str, dim: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # Trust the model's real dimension over the configured one.
        self.dim = self.model.get_sentence_embedding_dimension() or dim

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts, batch_size=32, show_progress_bar=len(texts) > 64,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        return vec.astype(np.float32)


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedding backend (requires GEMINI_API_KEY)."""

    def __init__(self, model_name: str, dim: int, api_key: str):
        if not api_key:
            raise ValueError(
                "Gemini embedding provider selected but no GEMINI_API_KEY is set."
            )
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.dim = dim

    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.models.embed_content(model=self.model_name, contents=texts)
        vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
        return vecs

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Batch to stay within request limits.
        out = [self._embed(texts[i:i + 100]) for i in range(0, len(texts), 100)]
        return np.vstack(out).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def get_embedder(config: Config) -> BaseEmbedder:
    """Factory: build the embedder for the configured provider."""
    if config.embedding_provider == "gemini":
        return GeminiEmbedder(
            config.embedding_model, config.embedding_dim, config.gemini_api_key
        )
    return LocalEmbedder(config.embedding_model, config.embedding_dim)
