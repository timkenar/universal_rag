"""Pluggable persistent-memory backends.

The RAG system's *durable* memory (facts learned across sessions) lives behind
this interface, mirroring the provider pattern already used for embeddings and
LLMs. Two backends ship:

- ``obsidian``    -> :class:`~core.obsidian_memory.ObsidianMemory`, an offline
  markdown vault whose notes are indexed into the shared FAISS + BM25 store and
  retrieved like any other document (``provides_index = True``).
- ``supermemory`` -> :class:`~core.supermemory_backend.SupermemoryBackend`, an
  external memory engine that holds facts in its own store and returns them via
  :meth:`recall` (``provides_index = False``).

This is intentionally separate from :class:`~core.memory.ConversationMemory`,
which is the *ephemeral* within-session chat window.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import Config


@dataclass
class MemoryHit:
    """A recalled memory returned by an external backend, ready for the prompt."""

    text: str
    score: float = 0.0
    source: str = "memory"


class BaseMemoryBackend(ABC):
    """Common interface for durable memory backends."""

    #: True  -> :meth:`remember` returns a file the pipeline chunks and adds to
    #:          the shared hybrid index; recall happens through normal retrieval.
    #: False -> the backend stores memories itself and serves them via
    #:          :meth:`recall`, which the pipeline injects into the prompt.
    provides_index: bool = False

    @abstractmethod
    def remember(
        self, question: str, answer: str, sources: Optional[List[str]] = None
    ) -> Optional[Path]:
        """Persist a memory.

        Returns the path of a written note for index-backed backends (or ``None``
        if it already exists / was deduplicated), and ``None`` for external
        backends that keep memories in their own store.
        """

    @abstractmethod
    def recall(self, query: str, k: int) -> List[MemoryHit]:
        """Return up to ``k`` relevant memories to inject into the prompt.

        Index-backed backends return ``[]`` — their memories surface through the
        shared retriever instead.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of stored memories (``-1`` if the backend can't report it)."""

    def list_notes(self) -> List[Path]:
        """Vault note paths, for index-backed backends. Empty otherwise."""
        return []

    def name(self) -> str:
        return self.__class__.__name__


def get_memory_backend(config: Config) -> Optional[BaseMemoryBackend]:
    """Factory: build the durable memory backend for the configured provider.

    ``MEMORY_PROVIDER`` selects it: ``obsidian`` (default) | ``supermemory`` |
    ``none``. Returns ``None`` when memory is disabled.
    """
    provider = (config.memory_provider or "none").strip().lower()
    if provider in ("none", "off", "disabled", ""):
        return None
    if provider == "obsidian":
        from core.obsidian_memory import ObsidianMemory

        return ObsidianMemory(config.vault_dir)
    if provider == "supermemory":
        from core.supermemory_backend import SupermemoryBackend

        return SupermemoryBackend(config)
    raise ValueError(
        f"Unknown MEMORY_PROVIDER '{provider}' "
        "(expected 'obsidian', 'supermemory', or 'none')."
    )
