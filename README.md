# Universal RAG System

A provider-swappable Retrieval-Augmented Generation system that runs **fully offline by
default** (no API key) and upgrades to Gemini or Ollama with a single config flag.

It follows the classic production RAG pipeline used by LlamaIndex / LangChain / Haystack:

```
files ──▶ loaders ──▶ chunking ──▶ embeddings ──┐
                                                 ├─▶ hybrid retrieval (RRF) ──▶ rerank ──▶ LLM ──▶ answer
             BM25 sparse index ──────────────────┘         ▲                                  ▲
                                                       query cache                    conversation memory
```

**Universal** in three senses:
1. **Any file type** — PDF, TXT/MD, HTML, CSV, DOCX, and source code, via a loader registry.
2. **Swappable providers** — local embeddings ↔ Gemini; extractive ↔ Gemini ↔ Ollama LLM.
3. **Offline-first** — defaults need no API key; everything runs on your machine.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# (optional) enable Gemini or Ollama providers
pip install google-genai      # for Gemini
pip install requests          # for Ollama
```

> **First run** downloads the local embedding model (`all-MiniLM-L6-v2`, ~90 MB) and the
> cross-encoder reranker (~80 MB) from Hugging Face. Needs internet **once**, then fully offline.

---

## Usage

```bash
# Index a single file or an entire folder (recursive)
python main.py ingest files/
python main.py ingest path/to/document.pdf

# Ask a one-shot question
python main.py query "What does the document say about X?"

# Interactive multi-turn chat (uses conversation memory)
python main.py chat

# Inspect the current index + providers
python main.py status
```

Indexes persist to `storage/index/` and the query cache to `storage/cache/`, so ingestion is
a one-time step — later `query`/`chat` runs load straight from disk.

### Voice agent (JARVIS HUD)

A browser voice HUD ([web/jarvis.html](web/jarvis.html)) drives the same pipeline by speech —
ask a question out loud, watch the ring go to *PARSING* while the pipeline retrieves, then hear
the answer spoken back with its sources listed under the reply in the transcript.

```bash
python main.py serve                          # http://localhost:8000  (JARVIS_PORT overrides)
# or, for auto-reload during development:
uvicorn api.server:app --reload --port 8000
# then open http://localhost:8000 in Chrome and allow the microphone
```

[api/server.py](api/server.py) is a thin adapter over `RAGPipeline` — it builds the pipeline
**once** at startup and runs the blocking `query` in a threadpool so concurrent requests don't
stall the event loop.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`  | `/`           | serve the JARVIS HUD |
| `POST` | `/api/ask`    | `{"question": str, "use_memory": bool}` → `{"text", "full", "cached", "sources": [{"filename", "chunk_index", "score"}]}` (spoken `text` capped at ~600 chars; `full` is untruncated) |
| `GET`  | `/api/status` | `pipeline.status()` |

On any pipeline error the API returns `{"text": "I lost the link to the index — try again.",
"error": true}` (HTTP 200) so the HUD stays in character. CORS is restricted to `http://localhost`
/ `http://127.0.0.1` (any port).

Speech-to-text and text-to-speech use the **browser's built-in Web Speech API** — no API keys,
no audio leaves the machine (STT works best in Chrome). Local UI voice-commands (focus mode,
panel toggles, clear log, mute, diagnostics) stay in the browser and never hit the server.

> **Offline fallback:** the HUD still opens as a plain file with no server — it falls back to its
> built-in reply table, and pressing **SPACE** runs a scripted demo. To point it at a different
> API, set `window.JARVIS_API = { url, enabled }` in the page.

---

## Testing

There is no automated test suite yet — verification is a manual smoke test in two layers: the
**CLI pipeline** first (retrieval → rerank → LLM), then the **HUD + API** on top. Everything below
runs fully offline on the defaults (`LLM_PROVIDER=none`), so no API key is needed.

### 0. One-time setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> The first `ingest`/`query` downloads the embedding + reranker models (~170 MB) from Hugging
> Face; it needs internet **once**, then runs offline. A sample document lives at
> [files/sample.md](files/sample.md) so you have something to query immediately.

### 1. CLI pipeline

```bash
python main.py ingest files/                              # index the sample doc
python main.py status                                     # dense_vectors should be > 0
python main.py query "How long can Project Aurora stay aloft?"
python main.py query "Who leads the program?"
python main.py chat                                       # then: "and where is the avionics team?"
```

**What confirms it works:**
- Each answer lists a `[sample.md]` source line — retrieval is grounding on the doc.
- *"How long can it stay aloft?"* surfaces the **45 days** fact even though the doc says
  "endurance," not "aloft" — that's semantic (dense) retrieval, not keyword matching.
- A `chat` follow-up that only makes sense given the previous turn (e.g. *"and where is the
  avionics team?"*) proves conversation memory is in play.

### 2. Spin up JARVIS (HUD + API)

```bash
python main.py serve                                      # http://localhost:8000
# JARVIS_PORT=9000 python main.py serve                   # override the port
# uvicorn api.server:app --reload --port 8000             # dev mode, auto-reload
```

Verify the API without a microphone (note: `use_memory` is optional, defaults to `true`):

```bash
curl localhost:8000/api/status
curl -X POST localhost:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"question":"What is the wingspan?"}'
```

A healthy `/api/ask` reply is JSON with `text` (spoken, capped at ~600 chars), `full`
(untruncated), `cached`, and a `sources` array. If the index is missing you get
`{"text":"I lost the link to the index — try again.","error":true}` — run step 1 first.

### 3. Full voice test

Open `http://localhost:8000` in **Chrome**, allow the microphone, and speak
*"What is the maximum endurance?"* You should see the question transcribed, the ring move to
*PARSING*, then hear the answer spoken back with its sources listed under the reply.

> Ingestion is **CLI-only** — the API exposes `/api/ask` and `/api/status` but no ingest
> endpoint, so add new documents with `python main.py ingest <path>` before querying them.

---

## Configuration

All knobs live in [config.py](config.py) and can be overridden by environment variables
(see [.env.example](.env.example) — copy it to `.env`).

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers, 384-d) or `gemini` (3072-d) |
| `LLM_PROVIDER` | `none` | `none` = extractive answers w/ citations; or `gemini` / `ollama` |
| `chunk_size` / `chunk_overlap` | 512 / 64 | characters |
| `top_k_dense` / `top_k_sparse` | 10 / 10 | candidates from each retriever |
| `top_k_hybrid` | 10 | kept after RRF fusion |
| `top_k_final` | 5 | kept after reranking |
| `rrf_k` | 60 | RRF fusion constant |
| `use_reranker` | `True` | cross-encoder rerank on/off |
| `memory_window` | 5 | turns kept in chat memory |

### Switching to Gemini (no code changes)

```bash
export GEMINI_API_KEY="your-key"
export EMBEDDING_PROVIDER=gemini
export LLM_PROVIDER=gemini
python main.py query "..."
```

> Note: changing the embedding provider changes the vector dimension, so **re-ingest** into a
> fresh `storage/index/` after switching (delete the folder or point `index_dir` elsewhere).

---

## Architecture

| Module | Responsibility |
|---|---|
| [config.py](config.py) | Central dataclass config, provider resolution, storage paths |
| [core/document_processor.py](core/document_processor.py) | Loaders (`LOADERS` registry) + `RecursiveChunker` → `Chunk`s |
| [core/embeddings.py](core/embeddings.py) | `LocalEmbedder` / `GeminiEmbedder` behind `BaseEmbedder` |
| [core/vector_store.py](core/vector_store.py) | FAISS `IndexFlatIP` dense store (cosine), persistence |
| [core/bm25_store.py](core/bm25_store.py) | BM25 sparse lexical store, persistence |
| [core/hybrid_search.py](core/hybrid_search.py) | Reciprocal Rank Fusion of dense + sparse |
| [core/reranker.py](core/reranker.py) | Cross-encoder re-ranking of the shortlist |
| [core/cache.py](core/cache.py) | Disk-backed query→answer cache (config-signature keyed) |
| [core/memory.py](core/memory.py) | Sliding-window conversation memory |
| [core/llm.py](core/llm.py) | `ExtractiveLLM` / `GeminiLLM` / `OllamaLLM` behind `BaseLLM` |
| [rag/pipelines.py](rag/pipelines.py) | `RAGPipeline` — wires everything together |
| [main.py](main.py) | CLI: `ingest` / `query` / `chat` / `status` / `serve` |
| [api/server.py](api/server.py) | FastAPI adapter: serves the HUD + `POST /api/ask`, `GET /api/status` |
| [web/jarvis.html](web/jarvis.html) | JARVIS voice HUD (Web Speech API, self-contained) |

### Extending

- **New file type** → add a loader function and register it in `LOADERS` in
  [core/document_processor.py](core/document_processor.py).
- **New embedding/LLM provider** → subclass `BaseEmbedder` / `BaseLLM` and add a branch to the
  respective `get_*` factory.
- **New vector store** (Chroma, Qdrant, …) → mirror the `VectorStore` interface (`add`, `search`,
  `save`, `load`) and swap it in `RAGPipeline.__init__`.

---

## How retrieval works

1. The query is embedded and searched against FAISS (**dense**, semantic) → top-k.
2. The query is tokenized and scored by BM25 (**sparse**, lexical/keyword) → top-k.
3. The two ranked lists are fused with **RRF**: each result adds `1 / (rrf_k + rank)` to its
   chunk's score, so a chunk ranked highly by *either* retriever floats up — no score
   calibration needed.
4. The fused shortlist is re-scored by a **cross-encoder** that reads (query, passage) jointly.
5. The top passages become the LLM context; the answer cites them as `[n]`.
