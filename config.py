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


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment ('1'/'true'/'yes'/'on' -> True)."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Per-provider embedding model + output dimensionality.
EMBEDDING_MODELS = {
    "local": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "gemini": ("models/gemini-embedding-001", 3072),
}

# Default base URLs for OpenAI-compatible LLM providers. Any endpoint that
# speaks the OpenAI Chat Completions API works here — OpenAI, NVIDIA
# (build.nvidia.com), Groq, Together, OpenRouter, a local vLLM/LM Studio, etc.
# Pick one with LLM_PROVIDER, or set LLM_PROVIDER=openai + OPENAI_BASE_URL for
# anything not listed.
OPENAI_COMPATIBLE_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass
class Config:
    """Runtime configuration. Instantiate with ``Config()`` for defaults."""

    # --- Provider selection -------------------------------------------------
    # embedding_provider: "local" | "gemini"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    # llm_provider: "none" | "gemini" | "anthropic" | "ollama" | OpenAI-compatible
    #   "none"      -> extractive answers with citations (works offline, no key)
    #   "anthropic" -> Claude via the official Anthropic SDK
    #   OpenAI-compatible presets: "openai" | "nvidia" | "groq" | "together" |
    #     "openrouter" (or "openai" + OPENAI_BASE_URL for any other endpoint)
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
    # Anthropic uses its own model id namespace; keep it separate from llm_model.
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # OpenAI-compatible endpoint override (leave blank to use the preset URL for
    # the selected llm_provider, e.g. NVIDIA when llm_provider="nvidia").
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    max_tokens: int = 4096
    temperature: float = 0.2

    # --- Conversation memory (ephemeral, within-session) --------------------
    memory_window: int = 5      # number of previous turns kept for context

    # --- Durable memory layer (persists across sessions) --------------------
    # memory_provider: "obsidian" (offline markdown vault, default) |
    #   "supermemory" (external engine) | "none"
    memory_provider: str = os.getenv("MEMORY_PROVIDER", "obsidian")
    # Auto-persist every answered query/chat as a memory (skipped in offline
    # extractive mode, whose answers just echo indexed documents).
    memory_autosave: bool = _env_bool("MEMORY_AUTOSAVE", True)
    # Memory hits injected into the prompt by external (non-indexed) backends.
    top_k_memory: int = 3
    # Obsidian vault directory (memory_provider="obsidian").
    vault_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("OBSIDIAN_VAULT_DIR") or (BASE_DIR / "memory_vault")
        )
    )
    # supermemory settings (memory_provider="supermemory").
    supermemory_api_key: str = field(
        default_factory=lambda: os.getenv("SUPERMEMORY_API_KEY", "")
    )
    supermemory_base_url: str = os.getenv("SUPERMEMORY_BASE_URL", "")
    supermemory_user_id: str = os.getenv("SUPERMEMORY_USER_ID", "default")

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
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    # Shared by all OpenAI-compatible providers. NVIDIA/Groq/etc. keys are read
    # here too (via their own env var, falling back to OPENAI_API_KEY).
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("TOGETHER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    )

    def resolve_openai_base_url(self) -> str:
        """Base URL for the selected OpenAI-compatible provider.

        An explicit ``openai_base_url`` (OPENAI_BASE_URL) always wins; otherwise
        fall back to the preset for the provider name, then to OpenAI itself.
        """
        if self.openai_base_url:
            return self.openai_base_url
        return OPENAI_COMPATIBLE_BASE_URLS.get(
            self.llm_provider, OPENAI_COMPATIBLE_BASE_URLS["openai"]
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

        # The offline vault lives on disk; create it up front.
        self.vault_dir = Path(self.vault_dir)
        if self.memory_provider == "obsidian":
            self.vault_dir.mkdir(parents=True, exist_ok=True)

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
