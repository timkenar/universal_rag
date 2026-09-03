"""External memory backend powered by supermemory (https://supermemory.ai).

supermemory is a dedicated memory engine (MIT-licensed) that adds automatic fact
extraction, temporal handling, contradiction resolution and expiry on top of raw
storage. Unlike the offline vault, it keeps memories in its own store and serves
them via :meth:`recall`, which the pipeline injects into the prompt as known
facts.

Deployment (via config / env):
  - Self-host: ``SUPERMEMORY_BASE_URL=http://localhost:6767`` (`npx supermemory
    local`) — data stays on your machine, but it runs as a separate service.
  - Cloud:     ``SUPERMEMORY_API_KEY=...`` against the hosted API — data leaves
    the machine, so this is opt-in only and warns at startup.

This adapter is written against supermemory's documented client surface
(``add`` / ``search``). The SDK is imported lazily so the dependency is only
needed when ``MEMORY_PROVIDER=supermemory``; install it with
``pip install supermemory``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional

from config import Config
from core.memory_backend import BaseMemoryBackend, MemoryHit


class SupermemoryBackend(BaseMemoryBackend):
    """Store/recall memories through the supermemory engine."""

    provides_index = False  # supermemory holds memories; recall() injects them

    def __init__(self, config: Config):
        self.config = config
        self.container = config.supermemory_user_id or "default"
        self._client: Any = None
        if not config.supermemory_api_key and not config.supermemory_base_url:
            warnings.warn(
                "MEMORY_PROVIDER=supermemory but neither SUPERMEMORY_API_KEY nor "
                "SUPERMEMORY_BASE_URL is set. Set a base URL for a self-hosted "
                "instance (npx supermemory local -> http://localhost:6767) or an "
                "API key for the cloud."
            )
        elif config.supermemory_api_key and not config.supermemory_base_url:
            warnings.warn(
                "supermemory cloud selected — memories will be sent off this "
                "machine. Set SUPERMEMORY_BASE_URL to keep them local."
            )

    @property
    def client(self) -> Any:
        """Lazily build the supermemory client (imported only when used)."""
        if self._client is None:
            try:
                from supermemory import Supermemory
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "MEMORY_PROVIDER=supermemory requires the 'supermemory' "
                    "package. Install it with: pip install supermemory"
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.config.supermemory_api_key:
                kwargs["api_key"] = self.config.supermemory_api_key
            if self.config.supermemory_base_url:
                kwargs["base_url"] = self.config.supermemory_base_url
            self._client = Supermemory(**kwargs)
        return self._client

    # --- Write --------------------------------------------------------------
    def remember(
        self, question: str, answer: str, sources: Optional[List[str]] = None
    ) -> Optional[Path]:
        content = f"Q: {question}\nA: {answer}".strip()
        metadata = {"sources": ", ".join(sources or [])} if sources else {}
        try:
            # Documented surface: client.add(content, container_tag=..., metadata=...)
            self.client.add(
                content, container_tag=self.container, metadata=metadata
            )
        except TypeError:
            # Older/newer SDKs vary in kwarg names; fall back to positional.
            self.client.add(content)
        except Exception as exc:  # never let memory-writing break a query
            warnings.warn(f"supermemory add failed: {exc}")
        return None  # external store — nothing local to index

    # --- Read ---------------------------------------------------------------
    def recall(self, query: str, k: int) -> List[MemoryHit]:
        try:
            resp = self.client.search(
                query, container_tag=self.container, limit=k
            )
        except TypeError:
            resp = self.client.search(query)
        except Exception as exc:
            warnings.warn(f"supermemory search failed: {exc}")
            return []
        return self._to_hits(resp, k)

    @staticmethod
    def _to_hits(resp: Any, k: int) -> List[MemoryHit]:
        """Normalize supermemory's response shape into MemoryHits."""
        # Responses vary by SDK version: an object with `.results`, a dict with
        # "results"/"memories", or a bare list. Handle them all defensively.
        items = resp
        for attr in ("results", "memories", "data"):
            if hasattr(resp, attr):
                items = getattr(resp, attr)
                break
            if isinstance(resp, dict) and attr in resp:
                items = resp[attr]
                break
        if not isinstance(items, list):
            return []

        hits: List[MemoryHit] = []
        for item in items[:k]:
            text = None
            for key in ("memory", "content", "text", "chunk", "summary"):
                if isinstance(item, dict) and item.get(key):
                    text = item[key]
                    break
                if hasattr(item, key) and getattr(item, key):
                    text = getattr(item, key)
                    break
            if not text:
                continue
            score = 0.0
            if isinstance(item, dict):
                score = float(item.get("score") or item.get("similarity") or 0.0)
            hits.append(MemoryHit(text=str(text), score=score, source="supermemory"))
        return hits

    def count(self) -> int:
        return -1  # external engine; count not cheaply available

    def name(self) -> str:
        return "supermemory"
