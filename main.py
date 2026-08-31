"""CLI entry point for the universal RAG system.

Usage:
    python main.py ingest <path>        # index a file or directory
    python main.py query "<question>"   # one-shot question
    python main.py chat                 # interactive multi-turn session
    python main.py status               # show index / provider info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import Config
from rag.pipelines import Answer, RAGPipeline

try:
    from rich.console import Console
    from rich.panel import Panel

    _console = Console()
except ImportError:  # rich is optional; degrade to plain print
    _console = None


def _print(msg: str) -> None:
    if _console:
        _console.print(msg)
    else:
        print(msg)


def _show_answer(answer: Answer) -> None:
    tag = " (cached)" if answer.cached else ""
    if _console:
        _console.print(Panel(answer.text, title=f"Answer{tag}", border_style="cyan"))
    else:
        print(f"\n=== Answer{tag} ===\n{answer.text}")
    if answer.sources:
        _print("\n[dim]Sources:[/dim]" if _console else "\nSources:")
        for i, r in enumerate(answer.sources, start=1):
            src = r.chunk.metadata.get("filename", "?")
            idx = r.chunk.metadata.get("chunk_index", "?")
            _print(f"  [{i}] {src} (chunk {idx})  score={r.score:.4f}")


def cmd_ingest(pipeline: RAGPipeline, args) -> None:
    path = Path(args.path)
    if not path.exists():
        _print(f"[red]Path not found:[/red] {path}" if _console else f"Path not found: {path}")
        sys.exit(1)
    _print(f"Ingesting {path} …")
    n = pipeline.ingest(path)
    total = len(pipeline.vector_store)
    _print(f"Added {n} chunks. Index now holds {total} chunks.")


def cmd_query(pipeline: RAGPipeline, args) -> None:
    answer = pipeline.query(args.question, use_memory=False)
    _show_answer(answer)


def cmd_chat(pipeline: RAGPipeline, args) -> None:
    _print("Interactive chat. Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer = pipeline.query(question, use_memory=True)
        _show_answer(answer)
        print()


def cmd_status(pipeline: RAGPipeline, args) -> None:
    status = pipeline.status()
    _print("RAG system status:")
    for key, value in status.items():
        _print(f"  {key:20s}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal RAG system CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a file or directory")
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a one-shot question")
    p_query.add_argument("question", help="The question to ask")
    p_query.set_defaults(func=cmd_query)

    p_chat = sub.add_parser("chat", help="Interactive multi-turn session")
    p_chat.set_defaults(func=cmd_chat)

    p_status = sub.add_parser("status", help="Show index / provider info")
    p_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = RAGPipeline(Config())
    args.func(pipeline, args)


if __name__ == "__main__":
    main()
