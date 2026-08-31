"""Central configuration for the universal RAG system.

Everything is provider-swappable via the ``*_provider`` fields. The defaults are
offline-first (local sentence-transformers, no API key required); switching to
Gemini is a matter of setting ``GEMINI_API_KEY`` and flipping the provider flags
below — no code changes needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; env vars still work without it
    pass

BASE_DIR = Path(__file__).parent


# Per-provider embedding model + output dimensionality.
EMBEDDING_MODELS = {
    "local": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "gemini": ("models/gemini-embedding-001", 3072),
}


@dataclass
class Config:
    """Runtime configuration. Instantiate with ``Config()`` for defaults."""

    # --- Provider selection -------------------------------------------------
    # embedding_provider: "local" | "gemini"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    # llm_provider: "none" | "gemini" | "ollama"
    #   "none"   -> extractive answers with citations (works offline, no key)
    llm_provider: str = os.getenv("LLM_PROVIDER", "none")

    # --- Embedding configuration (resolved from provider in __post_init__) --
    embedding_model: str = ""
    embedding_dim: int = 0

    # --- Chunking configuration --------------------------------------------
    chunk_size: int = 512       # characters per chunk
    chunk_overlap: int = 64     # characters of overlap between adjacent chunks

    # --- Retrieval configuration -------------------------------------------
    top_k_dense: int = 10       # FAISS dense results
    top_k_sparse: int = 10      # BM25 sparse results
    top_k_hybrid: int = 10      # results kept after RRF fusion
    top_k_final: int = 5        # results kept after re-ranking
    rrf_k: int = 60             # Reciprocal Rank Fusion constant

    # --- Re-ranking configuration ------------------------------------------
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # --- LLM configuration --------------------------------------------------
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_tokens: int = 4096
    temperature: float = 0.2

    # --- Conversation memory ------------------------------------------------
    memory_window: int = 5      # number of previous turns kept for context

    # --- Storage paths ------------------------------------------------------
    index_dir: Path = field(default_factory=lambda: BASE_DIR / "storage" / "index")
    cache_dir: Path = field(default_factory=lambda: BASE_DIR / "storage" / "cache")
    use_cache: bool = True

    # --- API keys (read from env) ------------------------------------------
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )

    def __post_init__(self) -> None:
        # Resolve embedding model/dim from the selected provider unless the
        # caller has overridden them explicitly.
        if not self.embedding_model or not self.embedding_dim:
            model, dim = EMBEDDING_MODELS.get(
                self.embedding_provider, EMBEDDING_MODELS["local"]
            )
            self.embedding_model = self.embedding_model or model
            self.embedding_dim = self.embedding_dim or dim

        # Ensure storage directories exist.
        self.index_dir = Path(self.index_dir)
        self.cache_dir = Path(self.cache_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def signature(self) -> str:
        """A short string identifying the config knobs that affect answers.

        Used to key the query cache so results are invalidated when the
        retrieval/generation setup changes.
        """
        return "|".join(
            str(x)
            for x in (
                self.embedding_provider,
                self.embedding_model,
                self.llm_provider,
                self.llm_model,
                self.top_k_final,
                self.rrf_k,
                self.use_reranker,
            )
        )
