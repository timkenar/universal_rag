"""Thin HTTP adapter that serves the JARVIS HUD and answers from ``RAGPipeline``.

This module owns no retrieval logic — it is a small FastAPI surface over the
existing :class:`RAGPipeline`. The pipeline (which loads the FAISS + BM25 indexes
from disk) is built exactly once at startup and reused across requests; the
blocking ``query`` call runs in a threadpool so concurrent requests do not stall
the event loop.

Run:
    uvicorn api.server:app --reload --port 8000     # dev
    python main.py serve                            # same app via the CLI

Then open http://localhost:8000 and speak a question answerable from an
ingested document.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from config import Config
from rag.pipelines import Answer, RAGPipeline

# web/jarvis.html sits alongside this package's parent (the project root).
BASE_DIR = Path(__file__).resolve().parent.parent
HUD_FILE = BASE_DIR / "web" / "jarvis.html"

# Spoken replies are truncated to keep speech synthesis snappy; the full text is
# always returned separately so the transcript can show everything.
SPOKEN_CHAR_CAP = 600


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Build the pipeline once — this loads the vector + BM25 indexes from disk.
    app.state.pipeline = RAGPipeline(Config())
    yield


app = FastAPI(title="JARVIS RAG Voice Agent", lifespan=lifespan)

# CORS: only browsers served from localhost (any port) may call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    use_memory: bool = True


def _serialize_sources(answer: Answer) -> list[dict[str, Any]]:
    """Flatten ``Answer.sources`` into the shape the HUD transcript expects."""
    return [
        {
            "filename": r.chunk.metadata.get("filename", "?"),
            "chunk_index": r.chunk.metadata.get("chunk_index", "?"),
            "score": round(float(r.score), 4),
            "memory": r.chunk.metadata.get("type") == "memory",
        }
        for r in answer.sources
    ]


@app.get("/")
def index() -> FileResponse:
    """Serve the JARVIS voice HUD."""
    return FileResponse(HUD_FILE)


@app.post("/api/ask")
async def ask(req: Request, body: AskRequest) -> JSONResponse:
    pipeline: RAGPipeline = req.app.state.pipeline
    question = body.question.strip()
    if not question:
        return JSONResponse({"text": "I didn't catch a question.", "cached": False, "sources": []})
    try:
        # pipeline.query is synchronous/blocking — keep the event loop free.
        answer: Answer = await run_in_threadpool(
            pipeline.query, question, use_memory=body.use_memory
        )
    except Exception:
        # Never leak a stack trace to the HUD — stay in character.
        return JSONResponse(
            {"text": "I lost the link to the index — try again.", "error": True}
        )

    full = answer.text or ""
    spoken = full if len(full) <= SPOKEN_CHAR_CAP else full[:SPOKEN_CHAR_CAP].rstrip() + "…"
    return JSONResponse(
        {
            "text": spoken,
            "full": full,
            "cached": answer.cached,
            "sources": _serialize_sources(answer),
        }
    )


@app.get("/api/status")
async def status(req: Request) -> JSONResponse:
    pipeline: RAGPipeline = req.app.state.pipeline
    try:
        return JSONResponse(await run_in_threadpool(pipeline.status))
    except Exception:
        return JSONResponse({"error": True})


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("JARVIS_PORT", "8000")))
