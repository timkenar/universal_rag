"""Offline Obsidian-vault memory backend.

Persists learned facts as Obsidian-compatible markdown notes (YAML frontmatter +
``[[wikilinks]]`` + ``#tags``) in a vault folder you can open and edit in
Obsidian. The pipeline indexes each note into the shared FAISS + BM25 store, so
memories resurface through normal hybrid retrieval — no separate lookup and no
data leaving the machine.

Notes are keyed by a slug of the question, so asking the same thing twice maps to
one file (dedup), which also keeps duplicate vectors out of the append-only FAISS
index.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.memory_backend import BaseMemoryBackend, MemoryHit

# Small stopword list so auto-tags/slugs stay meaningful without an NLP dep.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "has", "have",
    "had", "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "can", "could", "would", "should", "will", "shall", "may", "might",
    "this", "that", "these", "those", "it", "its", "as", "if", "then", "than",
    "so", "not", "no", "yes", "you", "your", "me", "my", "we", "our", "i",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _keywords(text: str, limit: int = 6) -> List[str]:
    """Ordered, de-duplicated content words — used for tags."""
    out: List[str] = []
    for tok in _WORD_RE.findall(text.lower()):
        if len(tok) < 3 or tok in _STOPWORDS or tok in out:
            continue
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def _slugify(text: str, max_len: int = 60) -> str:
    slug = "-".join(_WORD_RE.findall(text.lower()))[:max_len].strip("-")
    return slug or "memory"


def _wikilink(source: str) -> str:
    """Turn a source filename into an Obsidian wikilink to its note name."""
    return f"[[{Path(source).stem}]]"


class ObsidianMemory(BaseMemoryBackend):
    """Durable memory as a folder of Obsidian markdown notes."""

    provides_index = True

    def __init__(self, vault_dir: Path):
        self.vault_dir = Path(vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    # --- Write --------------------------------------------------------------
    def remember(
        self, question: str, answer: str, sources: Optional[List[str]] = None
    ) -> Optional[Path]:
        sources = [s for s in (sources or []) if s]
        # Slug + short hash of the full question: identical questions collapse to
        # one file (dedup); distinct questions never collide.
        digest = hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()[:8]
        path = self.vault_dir / f"{_slugify(question)}-{digest}.md"
        if path.exists():
            return None  # already remembered -> skip write AND re-index
        path.write_text(self._render(question, answer, sources), encoding="utf-8")
        return path

    def _render(self, question: str, answer: str, sources: List[str]) -> str:
        created = datetime.now().isoformat(timespec="seconds")
        tags = _keywords(question)
        src_names = [Path(s).name for s in sources]

        fm = [
            "---",
            f"created: {created}",
            "type: memory",
            f"tags: [{', '.join(tags)}]",
            f"sources: [{', '.join(src_names)}]",
            "---",
            "",
        ]
        body = [f"# {question.strip()}", ""]
        answer = (answer or "").strip()
        if answer and answer != question.strip():
            body += [answer, ""]
        body += ["## Provenance", f"- Learned on {created}"]
        if src_names:
            links = ", ".join(_wikilink(s) for s in src_names)
            body.append(f"- Sources: {links}")
        if tags:
            body += ["", "See also: " + " ".join(f"[[{t}]]" for t in tags)]
        return "\n".join(fm + body) + "\n"

    # --- Read ---------------------------------------------------------------
    def recall(self, query: str, k: int) -> List[MemoryHit]:
        # Vault notes live in the shared hybrid index; they surface through
        # normal retrieval, so there is nothing to inject separately here.
        return []

    def list_notes(self) -> List[Path]:
        return sorted(self.vault_dir.glob("*.md"))

    def count(self) -> int:
        return len(self.list_notes())

    def name(self) -> str:
        return "obsidian"
