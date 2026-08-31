"""Universal document ingestion: loaders + chunking.

Turns any supported file into a list of :class:`Chunk` objects ready for
embedding and indexing. New file types are added by registering a loader in
``LOADERS`` — everything downstream is format-agnostic.
"""
from __future__ import annotations

import csv
import hashlib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from config import Config


@dataclass
class Chunk:
    """A retrievable unit of text plus provenance metadata."""

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Stable unique id derived from content + source + position."""
        source = self.metadata.get("source", "")
        index = self.metadata.get("chunk_index", "")
        raw = f"{source}:{index}:{self.text}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Loaders: each returns (full_text, base_metadata). Base metadata is merged
# into every chunk produced from the file.
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> Tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), {"pages": len(pages)}


def _load_text(path: Path) -> Tuple[str, dict]:
    return path.read_text(encoding="utf-8", errors="ignore"), {}


def _load_html(path: Path) -> Tuple[str, dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        warnings.warn("beautifulsoup4 not installed; reading HTML as raw text.")
        return _load_text(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n"), {}


def _load_csv(path: Path) -> Tuple[str, dict]:
    rows = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.reader(f):
            rows.append(" | ".join(row))
    return "\n".join(rows), {"rows": len(rows)}


def _load_docx(path: Path) -> Tuple[str, dict]:
    try:
        import docx
    except ImportError:
        warnings.warn("python-docx not installed; cannot read .docx. Skipping.")
        return "", {}
    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs)
    return text, {}


# Extension -> loader. Plain-text-ish formats share the text loader.
_TEXT_EXTS = [
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".sh",
]

LOADERS: Dict[str, Callable[[Path], Tuple[str, dict]]] = {
    ".pdf": _load_pdf,
    ".html": _load_html,
    ".htm": _load_html,
    ".csv": _load_csv,
    ".docx": _load_docx,
    **{ext: _load_text for ext in _TEXT_EXTS},
}


class RecursiveChunker:
    """Split text on natural boundaries down to ``chunk_size`` with overlap.

    Tries paragraph, then line, then sentence, then word boundaries so chunks
    stay semantically coherent rather than cutting mid-sentence.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        pieces = self._split_recursive(text, self.SEPARATORS)
        return self._merge(pieces)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]
        sep = separators[0]
        rest = separators[1:]
        parts = text.split(sep) if sep else list(text)
        out: List[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                if part:
                    out.append(part)
            else:
                out.extend(self._split_recursive(part, rest))
        return out

    def _merge(self, pieces: List[str]) -> List[str]:
        """Greedily merge small pieces up to chunk_size, then apply overlap."""
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Start next chunk with a tail of the previous one for overlap.
                if self.chunk_overlap and chunks:
                    tail = chunks[-1][-self.chunk_overlap:]
                    current = f"{tail} {piece}".strip()
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks


class DocumentProcessor:
    """Loads files and produces :class:`Chunk` objects."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.chunker = RecursiveChunker(
            self.config.chunk_size, self.config.chunk_overlap
        )

    @staticmethod
    def supported_extensions() -> List[str]:
        return sorted(LOADERS.keys())

    def load_document(self, path: Path) -> Tuple[str, dict]:
        """Dispatch to the right loader; fall back to text for unknown types."""
        path = Path(path)
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            warnings.warn(
                f"No loader for '{path.suffix}'; reading '{path.name}' as text."
            )
            loader = _load_text
        text, meta = loader(path)
        meta = {"source": str(path), "filename": path.name, **meta}
        return text, meta

    def process_file(self, path: Path) -> List[Chunk]:
        text, base_meta = self.load_document(path)
        chunks: List[Chunk] = []
        for i, piece in enumerate(self.chunker.split(text)):
            meta = {**base_meta, "chunk_index": i}
            chunks.append(Chunk(text=piece, metadata=meta))
        return chunks

    def process_directory(self, directory: Path, recursive: bool = True) -> List[Chunk]:
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        chunks: List[Chunk] = []
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and not path.name.startswith("."):
                try:
                    chunks.extend(self.process_file(path))
                except Exception as exc:  # keep going on individual failures
                    warnings.warn(f"Failed to process {path}: {exc}")
        return chunks

    def process(self, path: Path) -> List[Chunk]:
        """Convenience: dispatch to file or directory processing."""
        path = Path(path)
        if path.is_dir():
            return self.process_directory(path)
        return self.process_file(path)
