"""Dense vector store backed by FAISS.

Uses an inner-product index over L2-normalized vectors, which is equivalent to
cosine similarity. Chunks are stored in a parallel list and persisted alongside
the FAISS index.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from core.document_processor import Chunk


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class VectorStore:
    """FAISS ``IndexFlatIP`` dense store with cosine similarity."""

    INDEX_FILE = "faiss.index"
    META_FILE = "faiss_chunks.pkl"

    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {vectors.shape[1]} != index dim {self.dim}."
            )
        self.index.add(_normalize(vectors))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        q = _normalize(query_vector.reshape(1, -1))
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return self.index.ntotal

    # --- Persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / self.INDEX_FILE))
        with (directory / self.META_FILE).open("wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: Path, dim: int) -> "VectorStore":
        directory = Path(directory)
        store = cls(dim)
        index_path = directory / cls.INDEX_FILE
        meta_path = directory / cls.META_FILE
        if index_path.exists() and meta_path.exists():
            store.index = store._faiss.read_index(str(index_path))
            with meta_path.open("rb") as f:
                store.chunks = pickle.load(f)
        return store
