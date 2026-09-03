# Implementation Plan — Persistent Memory Layer

Status: **implemented** — all three phases shipped as one system (offline
Obsidian vault + pluggable `MEMORY_PROVIDER` + supermemory backend). See the
"Memory layer" section of the [README](../README.md) for usage and
`tests/test_memory_layer.py` for verification.
Owner: —
Last updated: 2026-09-03

This plan adds a **durable, cross-session memory layer** to the Universal RAG
system. It ships in one on-brand phase (an offline **Obsidian markdown vault**
indexed into the existing hybrid retriever) and documents a clear upgrade path
to a **pluggable `MEMORY_PROVIDER`** that can swap in
[supermemory](https://github.com/supermemoryai/supermemory) for stronger,
automatic memory reasoning.

---

## 1. Motivation & architecture fit

Today the system's "memory" is stateless across runs:

- [`core/memory.py`](../core/memory.py) — `ConversationMemory`, a
  `deque(maxlen=5)` that dies when the process exits.
- [`core/cache.py`](../core/cache.py) — verbatim question→answer cache only.

Retrieval only ever grounds on **ingested documents**. Facts established in a
chat are forgotten the moment the session ends.

The memory layer adds a **fourth knowledge tier** that persists and compounds,
ranked by RRF alongside the source documents:

```
                       ┌─ documents (PDF/MD/…)   ← what you ingested
 hybrid retrieval ─────┼─ memory notes (vault)   ← what the agent learned   ← NEW
                       │
 prompt-time only ─────┼─ conversation memory    (this session, ephemeral)
                       └─ query cache             (verbatim repeats)
```

**Why it fits cleanly (additive, no breaking changes):**

1. `.md` is already a supported loader (`_TEXT_EXTS` in
   [`core/document_processor.py`](../core/document_processor.py)) — an Obsidian
   note is an ingestible document with zero new parsing.
2. `VectorStore.add()` / `BM25Store.add()` accept arbitrary `Chunk`s — they
   don't care whether a chunk came from a PDF or a memory note.
3. `core/memory.py` is already the "memory" seam; the vault is its **durable
   sibling**, sitting right next to the ephemeral window.

**Design constraint to respect:** FAISS `IndexFlatIP`
([`core/vector_store.py`](../core/vector_store.py)) has **no delete**. Naive
re-indexing of the same memory duplicates vectors. The plan handles this with
content-slug dedup (Phase 1) and a documented "rebuild to prune" story.

---

## 2. Phase 1 — Offline Obsidian memory vault (recommended first step)

Goal: the agent writes salient chat answers as Obsidian-compatible markdown
notes; those notes are embedded into the **same** FAISS + BM25 index so they
resurface in future sessions, ranked alongside documents. Pure Python, offline,
no new services, human-editable in Obsidian.

Effort: **~1 new file + ~4 small edits.** Reversible, safe default-on (an empty
vault changes nothing about retrieval).

### 2.1 New module — `core/obsidian_memory.py`

A **pure vault manager** (no embedder/store coupling — the pipeline owns
indexing). Proposed API:

```python
class ObsidianMemory:
    def __init__(self, vault_dir: Path): ...

    def remember(self, question: str, answer: str,
                 sources: list[str] | None = None) -> Path | None:
        """Render + write an Obsidian note. Returns the path, or None if a note
        for this question already exists (dedup — caller then skips indexing)."""

    def list_notes(self) -> list[Path]: ...
    def __len__(self) -> int: ...          # note count, for status()
```

Behavior:

- **Filename** = content-slug of the question (lowercased, stopword-trimmed,
  hyphenated), **no timestamp**, so identical questions map to one file →
  natural dedup. If the file exists, `remember()` returns `None` and the caller
  does not re-index (prevents duplicate FAISS vectors).
- **Note format** — Obsidian-native YAML frontmatter + wikilinks:

  ```markdown
  ---
  created: 2026-09-03T10:45:00
  type: memory
  tags: [aurora, endurance]
  sources: [sample.md]
  ---

  # How long can Project Aurora stay aloft?

  Project Aurora has a maximum endurance of 45 days on station.

  ## Provenance
  - Learned during a chat on 2026-09-03
  - Sources: [[sample.md]]

  See also: [[project-aurora]]
  ```

- **Tags / wikilinks** — lightweight heuristic extraction: stopword-filtered
  keywords from the question for `tags`; source filenames become `[[note]]`
  wikilinks. Deliberately **no LLM call** (keeps it offline + fast). Weak tag
  quality is an accepted v1 tradeoff (see §5).

### 2.2 `config.py` additions

```python
use_obsidian_memory: bool  = env "OBSIDIAN_MEMORY"      default True
memory_autosave:     bool  = default True   # persist chat turns automatically
vault_dir:           Path  = env "OBSIDIAN_VAULT_DIR"   default BASE_DIR/"memory_vault"
```

`__post_init__` creates `vault_dir` when `use_obsidian_memory` is on.
No change to `signature()` (memory does not alter the query-cache key).

### 2.3 `rag/pipelines.py` wiring

- Construct `self.memory_vault = ObsidianMemory(cfg.vault_dir)` when enabled,
  else `None`.
- New method:

  ```python
  def remember(self, question, answer_text, sources=None) -> Path | None:
      if not self.memory_vault:
          return None
      note = self.memory_vault.remember(question, answer_text, sources)
      if note is None:              # dedup — already known & indexed
          return None
      chunks = self.processor.process_file(note)
      for c in chunks:
          c.metadata["type"] = "memory"     # label for source display + filtering
      if chunks:
          vectors = self.embedder.embed_documents([c.text for c in chunks])
          self.vector_store.add(chunks, vectors)
          self.bm25_store.add(chunks)
          self._persist()
      return note
  ```

  This reuses the **exact** ingest path — no new indexing logic.
- In `query()`: when `use_memory=True` and `cfg.memory_autosave`, call
  `self.remember(question, answer_text, [r.chunk.metadata.get("filename") for r in results])`
  after generation.
- `status()`: add `"memory_notes": len(self.memory_vault) if self.memory_vault else 0`.

### 2.4 `main.py` surface

- New `remember "<fact>"` subcommand to manually seed a memory (writes + indexes
  via `pipeline.remember`).
- `_show_answer`: label results whose `metadata["type"] == "memory"` as
  `[memory]` instead of a filename, so users can see when an answer is grounded
  on learned memory vs a source doc.
- `status`: print the new `memory_notes` count.

### 2.5 Supporting changes

- `.gitignore`: add `memory_vault/` (mirrors `storage/`). *(Or keep it tracked
  if the user wants memories in version control — decision point.)*
- `.env.example`: document `OBSIDIAN_MEMORY`, `OBSIDIAN_VAULT_DIR`,
  `memory_autosave`.
- `README.md`: add a "Memory layer" section + the fifth knowledge tier in the
  architecture diagram; note the "rebuild index to prune memories" caveat.

### 2.6 Verification (manual smoke test — offline, matches existing style)

```bash
python main.py ingest files/
python main.py chat            # ask a question, then exit
ls memory_vault/               # → a .md note with frontmatter + wikilinks
python main.py status          # → memory_notes: 1
python main.py remember "The avionics team is based in Nairobi."
python main.py query "Where is the avionics team?"   # answer cites a [memory] source
# Re-run the same chat question → NO second note, NO duplicate vectors (dedup works)
```

**Pass criteria:** note file is valid Obsidian (opens in a vault, wikilinks
resolve); a fact from a prior session resurfaces in a later `query` with a
`[memory]` source line; repeating a question does not grow the index.

---

## 3. Phase 2 — Retrieval & hygiene refinements (optional, still fully local)

Incremental, only if Phase 1 proves useful. Each is independent.

- **Memory-aware ranking** — small RRF boost/penalty for `type == "memory"`
  chunks (config knob `memory_rank_weight`) so learned facts don't drown out, or
  don't overwhelm, source docs.
- **Prune / forget** — because FAISS can't delete: a `memory rebuild` command
  that drops all `type == "memory"` chunks and re-indexes the current vault
  (which the user has edited/pruned in Obsidian). This is the supported
  "forget" path.
- **Backlink graph** — generate topic notes (`[[project-aurora]]`) and populate
  their backlinks so the vault becomes a real Obsidian graph over time.
- **Salience gate** — only auto-save when the answer is non-trivial (length /
  had sources / not "I don't know"), to keep the vault signal-dense.
- **API + JARVIS surface** — expose memory count in `/api/status` and optionally
  a "remembered that" cue in the HUD.

---

## 4. Phase 3 (future) — Pluggable `MEMORY_PROVIDER` + supermemory backend

The upgrade path if heuristic tagging/dedup/forgetting proves too weak.
supermemory ([MIT](https://github.com/supermemoryai/supermemory), `pip install
supermemory`) adds automatic **fact extraction, temporal handling, contradiction
resolution, expiry, and profiles** — benchmarked #1 on LongMemEval / LoCoMo /
ConvoMem. The goal is to gain that **without** losing offline-first as the
default.

### 4.1 Introduce a memory seam (mirrors `LLM_PROVIDER` / `get_llm`)

```python
class BaseMemoryBackend(ABC):
    def remember(self, question, answer, sources) -> None: ...
    def recall(self, query, k) -> list[MemoryHit]: ...   # facts to inject/rank
    def count(self) -> int: ...

def get_memory_backend(config) -> BaseMemoryBackend | None:
    # "obsidian" (default) | "supermemory" | "none"
```

- `config.memory_provider = env "MEMORY_PROVIDER" default "obsidian"`.
- `ObsidianMemory` becomes the default `BaseMemoryBackend` implementation
  (Phase 1 code, lightly adapted).
- `RAGPipeline` depends only on the interface, not the concrete class.

### 4.2 `SupermemoryBackend` adapter

- `remember()` → `client.add(...)`; `recall()` → `client.search(...)` /
  `client.profile(...)`.
- Deployment via config:
  - **Self-host** — `SUPERMEMORY_BASE_URL=http://localhost:6767` (`npx
    supermemory local`); keeps data on-machine but adds a separate service.
  - **Cloud** — `SUPERMEMORY_API_KEY` against `console.supermemory.ai`
    (breaks offline-first; data leaves the machine — must be explicit opt-in
    with a startup warning).
- Two integration modes for retrieval:
  1. **Prompt-inject** — `recall()` results prepended as "known facts" (simplest;
     bypasses local FAISS/BM25).
  2. **Index-mirror** — write supermemory-extracted facts back into the vault +
     local index so they still rank via RRF (keeps one retrieval path).

### 4.3 Migration, rollback, safety

- **Default unchanged:** with `MEMORY_PROVIDER` unset, behavior is identical to
  Phase 1 — no network, no new deps (supermemory SDK imported lazily, like the
  `google-genai` / `anthropic` / `openai` optional imports already are).
- **Rollback:** flip `MEMORY_PROVIDER=obsidian`; the markdown vault is the
  durable source of truth, so nothing is lost.
- **Offline guarantee:** the cloud path must never be reachable unless the user
  sets a key; log a clear "memory is leaving this machine" warning at startup.
- **Docs:** README gains a "Choosing a memory provider" section paralleling
  "Choosing an LLM provider," including the offline/self-host/cloud tradeoff
  table.

---

## 5. Risks, tradeoffs & scope boundaries

| Item | Phase | Mitigation |
|---|---|---|
| FAISS has no delete → duplicate vectors on re-index | 1 | Content-slug dedup; `memory rebuild` command (Phase 2) is the forget path |
| Heuristic tags/wikilinks are low quality | 1 | Accepted for v1; LLM-assisted extraction or supermemory (Phase 3) upgrades it |
| Vault could accumulate low-value notes | 1→2 | Salience gate (Phase 2); user prunes in Obsidian + rebuild |
| supermemory self-host adds a non-Python service | 3 | Optional, opt-in; default stays in-process |
| supermemory cloud sends data off-machine | 3 | Explicit key required + startup warning; never default |
| Memory notes could dominate retrieval | 2 | `memory_rank_weight` knob |

**Explicitly out of scope for v1 (Phase 1):** automatic contradiction
resolution, temporal/expiry logic, LLM-based fact extraction, and memory
deletion — all deferred to Phase 2/3.

---

## 6. Recommended sequencing

1. **Ship Phase 1.** Small, reversible, on-brand; delivers ~80% of the value
   (durable cross-session recall ranked with docs).
2. **Add Phase 2 items on demand** — only the ones real usage shows you need
   (most likely `memory rebuild` + salience gate first).
3. **Reach for Phase 3 / supermemory** only if heuristic memory reasoning proves
   insufficient — and even then, keep the offline Obsidian vault as the default
   backend and source of truth.
