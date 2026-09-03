"""Tests for the durable memory layer.

Two tiers so you can run *something* without the ML stack:

- ``VaultTests`` exercise the offline Obsidian backend + factory using only the
  standard library — they always run.
- ``PipelineMemoryTests`` exercise the full pipeline wiring (remember, recall,
  retrieval of memories, and the FAISS rebuild/prune path). They need
  ``numpy`` + ``faiss`` + ``rank_bm25`` but NOT torch/sentence-transformers: a
  stub embedder stands in, the reranker is off, and the LLM is the offline
  extractive one. They skip automatically if those deps are missing.

Run:
    python -m unittest tests.test_memory_layer -v
    # or just the always-on tier:
    python -m unittest tests.test_memory_layer.VaultTests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Make the project root importable when run as `python -m unittest`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from core.memory_backend import get_memory_backend
from core.obsidian_memory import ObsidianMemory

try:
    import numpy as np  # noqa: F401
    import faiss  # noqa: F401
    import rank_bm25  # noqa: F401

    _HEAVY = True
except ImportError:
    _HEAVY = False


class VaultTests(unittest.TestCase):
    """Offline Obsidian backend — stdlib only, always runs."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.mem = ObsidianMemory(self.tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_note_is_valid_obsidian_markdown(self) -> None:
        note = self.mem.remember(
            "How long can Project Aurora stay aloft?",
            "It has a maximum endurance of 45 days.",
            ["files/sample.md"],
        )
        self.assertIsNotNone(note)
        text = note.read_text()
        # Frontmatter, type marker, the answer, and a source wikilink.
        self.assertTrue(text.startswith("---\n"))
        self.assertIn("type: memory", text)
        self.assertIn("45 days", text)
        self.assertIn("[[sample]]", text)
        self.assertIn("tags: [", text)

    def test_duplicate_question_is_deduped(self) -> None:
        q = "Where is the avionics team?"
        first = self.mem.remember(q, "Nairobi.", [])
        second = self.mem.remember(q, "A different answer.", [])
        self.assertIsNotNone(first)
        self.assertIsNone(second, "same question must map to one note (dedup)")
        self.assertEqual(self.mem.count(), 1)

    def test_distinct_questions_do_not_collide(self) -> None:
        self.mem.remember("What is the wingspan?", "35 metres.", [])
        self.mem.remember("What is the endurance?", "45 days.", [])
        self.assertEqual(self.mem.count(), 2)

    def test_factory_selects_backend(self) -> None:
        cfg = Config(memory_provider="obsidian", vault_dir=self.tmp)
        backend = get_memory_backend(cfg)
        self.assertIsInstance(backend, ObsidianMemory)
        self.assertTrue(backend.provides_index)
        self.assertIsNone(get_memory_backend(Config(memory_provider="none")))
        with self.assertRaises(ValueError):
            get_memory_backend(Config(memory_provider="bogus"))


@unittest.skipUnless(_HEAVY, "needs numpy + faiss + rank_bm25 (no torch required)")
class PipelineMemoryTests(unittest.TestCase):
    """Full pipeline wiring with a stub embedder (no torch/model download)."""

    def _pipeline(self, tmp: Path):
        import numpy as np

        import rag.pipelines as pipelines
        from core.embeddings import BaseEmbedder

        class StubEmbedder(BaseEmbedder):
            """Deterministic hashing embedder — no model, no network."""

            dim = 32

            def _vec(self, text: str) -> np.ndarray:
                v = np.zeros(self.dim, dtype=np.float32)
                for tok in text.lower().split():
                    v[hash(tok) % self.dim] += 1.0
                return v

            def embed_documents(self, texts):
                if not texts:
                    return np.zeros((0, self.dim), dtype=np.float32)
                return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

            def embed_query(self, text):
                return self._vec(text)

        # Swap the real embedder factory for the stub for this pipeline build.
        original = pipelines.get_embedder
        pipelines.get_embedder = lambda config: StubEmbedder()
        try:
            cfg = Config(
                memory_provider="obsidian",
                vault_dir=tmp / "vault",
                index_dir=tmp / "index",
                cache_dir=tmp / "cache",
                use_reranker=False,   # skip the cross-encoder download
                llm_provider="none",  # offline extractive answers
                use_cache=False,
            )
            return pipelines.RAGPipeline(cfg)
        finally:
            pipelines.get_embedder = original

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.pipe = self._pipeline(self.tmp)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_remember_indexes_into_shared_store(self) -> None:
        before = len(self.pipe.vector_store)
        note = self.pipe.remember(
            "Where is the avionics team based?",
            "The avionics team is based in Nairobi.",
            [],
        )
        self.assertIsNotNone(note)
        self.assertGreater(
            len(self.pipe.vector_store), before,
            "a remembered note must add vectors to the shared index",
        )

    def test_memory_is_retrievable_and_labelled(self) -> None:
        self.pipe.remember(
            "Where is the avionics team based?",
            "The avionics team is based in Nairobi.",
            [],
        )
        answer = self.pipe.query("avionics team based located", use_memory=False)
        self.assertTrue(
            any(r.chunk.metadata.get("type") == "memory" for r in answer.sources),
            "the memory note should surface through hybrid retrieval",
        )

    def test_autosave_on_chat_turn(self) -> None:
        notes_before = self.pipe.memory_backend.count()
        self.pipe.query("What colour is the sky in the demo?", use_memory=True)
        self.assertEqual(
            self.pipe.memory_backend.count(), notes_before + 1,
            "a chat turn should auto-persist one memory note",
        )

    def test_rebuild_prunes_deleted_notes(self) -> None:
        # Two memories in the index...
        self.pipe.remember("Fact one about alpha?", "Alpha is one.", [])
        self.pipe.remember("Fact two about beta?", "Beta is two.", [])
        with_two = len(self.pipe.vector_store)
        # ...delete one note on disk (as a user would in Obsidian)...
        notes = self.pipe.memory_backend.list_notes()
        notes[0].unlink()
        # ...rebuild drops all memory vectors and re-indexes what's left.
        reindexed = self.pipe.rebuild_memory()
        self.assertEqual(reindexed, 1)
        self.assertLess(
            len(self.pipe.vector_store), with_two,
            "rebuild must remove vectors for the deleted memory note",
        )

    def test_rebuild_keeps_document_chunks(self) -> None:
        # Ingest the sample document, then a memory, then rebuild.
        sample = Path(__file__).resolve().parent.parent / "files" / "sample.md"
        if sample.exists():
            self.pipe.ingest(sample)
        doc_vectors = sum(
            1 for c in self.pipe.vector_store.chunks
            if c.metadata.get("type") != "memory"
        )
        self.pipe.remember("A remembered fact?", "Yes indeed.", [])
        self.pipe.rebuild_memory()
        kept_docs = sum(
            1 for c in self.pipe.vector_store.chunks
            if c.metadata.get("type") != "memory"
        )
        self.assertEqual(kept_docs, doc_vectors, "rebuild must preserve document chunks")


if __name__ == "__main__":
    unittest.main(verbosity=2)
