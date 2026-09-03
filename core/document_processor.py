"""Universal document ingestion: loaders + chunking.

Turns any supported file into a list of :class:`Chunk` objects ready for
embedding and indexing. New file types are added by registering a loader in
``LOADERS`` — everything downstream is format-agnostic.
"""
from __future__ import annotations

import csv
import hashlib
import os
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

# Minimum characters a page's text layer must have before we trust it and skip
# OCR. Scanned pages typically return "" or a few stray glyphs.
_OCR_MIN_CHARS = 20

# OCR toggle, read once from the environment so loaders (which take only a path)
# can see it. "auto" (default) OCRs only pages with no usable text layer;
# "off" disables OCR entirely; "force" OCRs every page.
_OCR_MODE = os.getenv("OCR_MODE", "auto").lower()


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from pdf2image import convert_from_path  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_image(image) -> str:
    """Run Tesseract on a PIL image; empty string on any failure."""
    try:
        import pytesseract

        return pytesseract.image_to_string(image) or ""
    except Exception as exc:  # tesseract binary missing, etc.
        warnings.warn(f"OCR failed: {exc}")
        return ""


def _load_pdf(path: Path) -> Tuple[str, dict]:
    """Extract the text layer; fall back to OCR on pages that lack one.

    Digital PDFs read instantly via pypdf. Scanned PDFs (image-only pages) have
    no text layer, so — unless OCR_MODE=off — each empty page is rendered to an
    image and passed through Tesseract. Set OCR_MODE=force to OCR every page.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]

    want_ocr = _OCR_MODE != "off"
    needs_ocr = [
        i for i, txt in enumerate(pages)
        if _OCR_MODE == "force" or len(txt.strip()) < _OCR_MIN_CHARS
    ]

    ocr_pages = 0
    if want_ocr and needs_ocr:
        if not _ocr_available():
            warnings.warn(
                f"'{path.name}' has {len(needs_ocr)} page(s) with no text layer "
                "(likely scanned), but OCR deps are missing. Install "
                "'pytesseract' + 'pdf2image' and the system 'tesseract-ocr' & "
                "'poppler' packages. Skipping those pages."
            )
        else:
            from pdf2image import convert_from_path

            # Render only the pages that need OCR (1-based page numbers).
            for i in needs_ocr:
                try:
                    images = convert_from_path(
                        str(path), first_page=i + 1, last_page=i + 1, dpi=300
                    )
                except Exception as exc:
                    warnings.warn(f"Could not render page {i + 1} of {path.name}: {exc}")
                    continue
                if images:
                    text = _ocr_image(images[0]).strip()
                    if text:
                        pages[i] = text
                        ocr_pages += 1

    return "\n\n".join(pages), {"pages": len(pages), "ocr_pages": ocr_pages}


def _load_image(path: Path) -> Tuple[str, dict]:
    """OCR a standalone image file (scanned page, screenshot, photo)."""
    if _OCR_MODE == "off" or not _ocr_available():
        warnings.warn(
            f"'{path.name}' is an image; OCR is required to read it. Install "
            "'pytesseract' + 'Pillow' and the system 'tesseract-ocr' package."
        )
        return "", {"ocr_pages": 0}
    from PIL import Image

    try:
        text = _ocr_image(Image.open(path)).strip()
    except Exception as exc:
        warnings.warn(f"Could not open image {path.name}: {exc}")
        return "", {"ocr_pages": 0}
    return text, {"ocr_pages": 1 if text else 0}


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

# Image formats read via OCR.
_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"]

LOADERS: Dict[str, Callable[[Path], Tuple[str, dict]]] = {
    ".pdf": _load_pdf,
    ".html": _load_html,
    ".htm": _load_html,
    ".csv": _load_csv,
    ".docx": _load_docx,
    **{ext: _load_image for ext in _IMAGE_EXTS},
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
