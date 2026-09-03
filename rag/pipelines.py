"""End-to-end RAG orchestration.

Wires together: document processing -> embeddings -> dense (FAISS) + sparse
(BM25) stores -> hybrid RRF retrieval -> cross-encoder rerank -> LLM generation,
with a query cache and conversation memory around it.
"""
from __future__ import annotations

import sys
import warnings
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
from core.memory_backend import get_memory_backend
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
        # Ephemeral within-session chat window ...
        self.memory = ConversationMemory(self.config.memory_window)
        # ... and the durable cross-session memory layer (may be None).
        self.memory_backend = get_memory_backend(self.config)
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
        # Standing context (identity/persona/pinned facts) injected into every
        # prompt's system message, regardless of the query — this is what makes an
        # identity persist across sessions rather than only surfacing when a
        # retrieval happens to match it.
        persistent = (
            self.memory_backend.persistent_context() if self.memory_backend else ""
        ).strip()

        # Cache only stateless queries, and only when no standing context applies
        # (identity influences the answer but isn't part of the cache key).
        use_query_cache = not use_memory and not persistent
        if use_query_cache:
            cached = self.cache.get(question)
            if cached is not None:
                return Answer(question, cached["text"], [], cached=True)

        results = self.retriever.retrieve(question)
        if self.reranker is not None:
            results = self.reranker.rerank(question, results, self.config.top_k_final)
        else:
            results = results[: self.config.top_k_final]

        contexts = [r.chunk.text for r in results]

        system = ""
        if persistent:
            system = (
                "Standing memory about this assistant and user — always honour "
                "it regardless of the retrieved context below:\n" + persistent
            )
        # Durable memory held outside the shared index (e.g. supermemory) is
        # pulled in here and prepended as known facts. The offline vault backend
        # returns nothing — its notes already rank inside `results` above.
        if self.memory_backend is not None:
            for hit in self.memory_backend.recall(question, self.config.top_k_memory):
                contexts.insert(0, hit.text)

        history = self.memory.format() if use_memory else ""
        answer_text = self.llm.generate(question, contexts, history, system=system)

        if use_memory:
            self.memory.add(question, answer_text)
            if self.config.memory_autosave:
                srcs = [r.chunk.metadata.get("filename", "") for r in results]
                try:
                    self.remember(question, answer_text, srcs)
                except Exception as exc:  # never let memory-writing break a query
                    warnings.warn(f"Could not persist memory: {exc}")
        elif use_query_cache:
            self.cache.set(question, {"text": answer_text})

        return Answer(question, answer_text, results)

    # --- Durable memory -----------------------------------------------------
    def remember(
        self, question: str, answer_text: str, sources: Optional[List[str]] = None
    ) -> Optional[Path]:
        """Persist a memory via the configured backend.

        For index-backed backends (the offline vault) the written note is chunked
        and added to the shared FAISS + BM25 store so it surfaces in future
        retrievals. Returns the note path, or ``None`` (disabled / duplicate /
        external backend).
        """
        backend = self.memory_backend
        if backend is None:
            return None
        note = backend.remember(question, answer_text, sources or [])
        if note is not None and backend.provides_index:
            self._index_memory_note(note)
        return note

    def set_identity(self, text: str) -> Optional[Path]:
        """Set the always-on identity/persona injected into every prompt."""
        if self.memory_backend is None:
            return None
        return self.memory_backend.pin(text, "identity")

    def clear_identity(self) -> bool:
        return self.memory_backend.clear_identity() if self.memory_backend else False

    def identity(self) -> str:
        """The current standing context (identity + any pinned facts)."""
        return self.memory_backend.persistent_context() if self.memory_backend else ""

    def _index_memory_note(self, note: Path) -> None:
        """Chunk a vault note and add it to the shared hybrid index."""
        chunks = self.processor.process_file(Path(note))
        if not chunks:
            return
        for chunk in chunks:
            chunk.metadata["type"] = "memory"
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.add(chunks, vectors)
        self.bm25_store.add(chunks)
        self._persist()

    def rebuild_memory(self) -> int:
        """Re-index the vault after edits/deletions in Obsidian (the prune path).

        FAISS ``IndexFlatIP`` has no per-vector delete, so this rebuilds the dense
        + sparse stores keeping only non-memory chunks, then re-indexes every
        current vault note. Returns the number of notes re-indexed. No-op for
        external backends. """
        backend = self.memory_backend
        if backend is None or not backend.provides_index:
            return 0
        self._drop_memory_chunks()
        notes = backend.list_notes()
        for note in notes:
            self._index_memory_note(note)
        self._persist()
        return len(notes)

    def _drop_memory_chunks(self) -> None:
        """Rebuild dense + sparse stores keeping only non-memory chunks."""
        import numpy as np

        store = self.vector_store
        keep = [i for i, c in enumerate(store.chunks)
                if c.metadata.get("type") != "memory"]
        kept_chunks = [store.chunks[i] for i in keep]

        fresh = VectorStore(self.embedder.dim)
        if keep and store.index.ntotal:
            # IndexFlatIP supports reconstruct: pull the kept (normalized) vectors
            # back out and re-add them to a fresh index.
            all_vecs = np.asarray(store.index.reconstruct_n(0, store.index.ntotal))
            fresh.add(kept_chunks, all_vecs[keep])
        self.vector_store = fresh

        fresh_bm = BM25Store()
        fresh_bm.add(kept_chunks)
        self.bm25_store = fresh_bm

        # Rewire the retriever to the fresh stores.
        self.retriever = HybridRetriever(
            self.embedder, self.vector_store, self.bm25_store, self.config
        )

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
            "memory_provider": self.memory_backend.name() if self.memory_backend else "none",
            "memory_notes": self.memory_backend.count() if self.memory_backend else 0,
            "identity_set": self.memory_backend.has_identity() if self.memory_backend else False,
        }
