"""Disk-backed query cache.

Keys answers by a hash of the query plus a config signature, so cached results
are automatically invalidated when the retrieval/generation setup changes. Uses
``diskcache`` when available, falling back to stdlib ``shelve``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional


class QueryCache:
    def __init__(self, cache_dir: Path, signature: str = "", enabled: bool = True):
        self.enabled = enabled
        self.signature = signature
        self._backend = None
        self._shelf_path: Optional[str] = None
        if not enabled:
            return

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import diskcache

            self._backend = diskcache.Cache(str(cache_dir / "querycache"))
        except ImportError:
            import shelve

            self._shelf_path = str(cache_dir / "querycache.shelf")
            self._backend = shelve.open(self._shelf_path)

    def _key(self, query: str) -> str:
        raw = f"{self.signature}::{query.strip().lower()}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        if not self.enabled or self._backend is None:
            return None
        return self._backend.get(self._key(query))

    def set(self, query: str, value: Any) -> None:
        if not self.enabled or self._backend is None:
            return
        self._backend[self._key(query)] = value

    def close(self) -> None:
        if self._backend is not None and hasattr(self._backend, "close"):
            self._backend.close()
