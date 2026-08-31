"""Sparse lexical store backed by BM25 (rank_bm25).

Complements the dense vector store: BM25 excels at exact keyword / rare-term
matches that embeddings can miss. Results are fused later via RRF.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Tuple

from core.document_processor import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """In-memory BM25Okapi index over chunk text."""

    STORE_FILE = "bm25.pkl"

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._tokens: List[List[str]] = []
        self._bm25 = None  # lazily (re)built when needed

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._tokens.extend(tokenize(c.text) for c in chunks)
        self._bm25 = None  # invalidate; rebuild on next search

    def _ensure_index(self) -> None:
        if self._bm25 is None and self._tokens:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        self._ensure_index()
        if self._bm25 is None or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        k = min(k, len(self.chunks))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self.chunks)

    # --- Persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Persist tokens + chunks (not the BM25 object) so it rebuilds cleanly.
        with (directory / self.STORE_FILE).open("wb") as f:
            pickle.dump({"chunks": self.chunks, "tokens": self._tokens}, f)

    @classmethod
    def load(cls, directory: Path) -> "BM25Store":
        store = cls()
        path = Path(directory) / cls.STORE_FILE
        if path.exists():
            with path.open("rb") as f:
                data = pickle.load(f)
            store.chunks = data["chunks"]
            store._tokens = data["tokens"]
        return store
