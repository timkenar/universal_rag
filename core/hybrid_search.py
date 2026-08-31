"""Hybrid retrieval: dense + sparse fused with Reciprocal Rank Fusion (RRF).

RRF combines rankings without needing to calibrate the very different score
scales of cosine similarity vs BM25. Each result contributes ``1/(rrf_k + rank)``
to its chunk's fused score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from config import Config
from core.bm25_store import BM25Store
from core.document_processor import Chunk
from core.embeddings import BaseEmbedder
from core.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float          # fused RRF score
    dense_rank: int = -1  # 0-based rank in dense results (-1 if absent)
    sparse_rank: int = -1


class HybridRetriever:
    """Fuse FAISS dense results and BM25 sparse results via RRF."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        config: Config,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.config = config

    def retrieve(self, query: str) -> List[RetrievalResult]:
        cfg = self.config

        qvec = self.embedder.embed_query(query)
        dense = self.vector_store.search(qvec, cfg.top_k_dense)
        sparse = self.bm25_store.search(query, cfg.top_k_sparse)

        # chunk_id -> aggregated result
        fused: Dict[str, RetrievalResult] = {}

        def _fuse(results: List[Tuple[Chunk, float]], which: str) -> None:
            for rank, (chunk, _score) in enumerate(results):
                cid = chunk.chunk_id
                entry = fused.get(cid)
                if entry is None:
                    entry = RetrievalResult(chunk=chunk, score=0.0)
                    fused[cid] = entry
                entry.score += 1.0 / (cfg.rrf_k + rank)
                if which == "dense":
                    entry.dense_rank = rank
                else:
                    entry.sparse_rank = rank

        _fuse(dense, "dense")
        _fuse(sparse, "sparse")

        ranked = sorted(fused.values(), key=lambda r: r.score, reverse=True)
        return ranked[: cfg.top_k_hybrid]
