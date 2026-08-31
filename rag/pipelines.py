"""End-to-end RAG orchestration.

Wires together: document processing -> embeddings -> dense (FAISS) + sparse
(BM25) stores -> hybrid RRF retrieval -> cross-encoder rerank -> LLM generation,
with a query cache and conversation memory around it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Allow running as `python rag/pipelines.py` as well as importing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from core.bm25_store import BM25Store
from core.cache import QueryCache
from core.document_processor import Chunk, DocumentProcessor
from core.embeddings import get_embedder
from core.hybrid_search import HybridRetriever, RetrievalResult
from core.llm import get_llm
from core.memory import ConversationMemory
from core.reranker import get_reranker
from core.vector_store import VectorStore


@dataclass
class Answer:
    question: str
    text: str
    sources: List[RetrievalResult]
    cached: bool = False


class RAGPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.processor = DocumentProcessor(self.config)
        self.embedder = get_embedder(self.config)

        # Load persisted indexes if present, else start empty.
        self.vector_store = VectorStore.load(
            self.config.index_dir, self.embedder.dim
        )
        self.bm25_store = BM25Store.load(self.config.index_dir)

        self.retriever = HybridRetriever(
            self.embedder, self.vector_store, self.bm25_store, self.config
        )
        self.reranker = get_reranker(self.config)
        self.llm = get_llm(self.config)
        self.memory = ConversationMemory(self.config.memory_window)
        self.cache = QueryCache(
            self.config.cache_dir, self.config.signature(), self.config.use_cache
        )

    # --- Ingestion ----------------------------------------------------------
    def ingest(self, path: Path) -> int:
        """Process a file or directory and add it to both indexes."""
        chunks: List[Chunk] = self.processor.process(Path(path))
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.add(chunks, vectors)
        self.bm25_store.add(chunks)
        self._persist()
        return len(chunks)

    def _persist(self) -> None:
        self.vector_store.save(self.config.index_dir)
        self.bm25_store.save(self.config.index_dir)

    # --- Querying -----------------------------------------------------------
    def query(self, question: str, use_memory: bool = False) -> Answer:
        # Cache only stateless (non-conversational) queries.
        if not use_memory:
            cached = self.cache.get(question)
            if cached is not None:
                return Answer(question, cached["text"], [], cached=True)

        results = self.retriever.retrieve(question)
        if self.reranker is not None:
            results = self.reranker.rerank(question, results, self.config.top_k_final)
        else:
            results = results[: self.config.top_k_final]

        contexts = [r.chunk.text for r in results]
        history = self.memory.format() if use_memory else ""
        answer_text = self.llm.generate(question, contexts, history)

        if use_memory:
            self.memory.add(question, answer_text)
        else:
            self.cache.set(question, {"text": answer_text})

        return Answer(question, answer_text, results)

    # --- Introspection ------------------------------------------------------
    def status(self) -> dict:
        return {
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "embedding_dim": self.embedder.dim,
            "llm_provider": self.config.llm_provider,
            "dense_vectors": len(self.vector_store),
            "sparse_docs": len(self.bm25_store),
            "reranker": bool(self.reranker),
            "index_dir": str(self.config.index_dir),
        }
