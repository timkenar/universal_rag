"""Cross-encoder re-ranking of retrieved candidates.

A cross-encoder scores each (query, passage) pair jointly, which is far more
accurate than the bi-encoder retrieval scores — but too slow to run over the
whole corpus, so it only re-ranks the hybrid shortlist. Loaded lazily; if the
model can't be loaded, retrieval order is passed through unchanged.
"""
from __future__ import annotations

import warnings
from typing import List

from config import Config
from core.hybrid_search import RetrievalResult


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._failed = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:  # missing model/weights/network
            warnings.warn(f"Reranker unavailable ({exc}); passing through.")
            self._failed = True
            return False

    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        if not results:
            return []
        if not self._ensure_model():
            return results[:top_k]

        pairs = [[query, r.chunk.text] for r in results]
        scores = self._model.predict(pairs)
        for r, s in zip(results, scores):
            r.score = float(s)  # overwrite fused score with rerank score
        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        return ranked[:top_k]


def get_reranker(config: Config) -> CrossEncoderReranker | None:
    if not config.use_reranker:
        return None
    return CrossEncoderReranker(config.rerank_model)
