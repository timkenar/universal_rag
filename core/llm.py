"""Answer generation behind a swappable interface.

Providers:
  * ``none``   -> ExtractiveLLM: no API key needed; stitches the top passages
                  into an answer with inline [n] citations. Works fully offline.
  * ``gemini`` -> GeminiLLM: Google Gemini (requires GEMINI_API_KEY).
  * ``ollama`` -> OllamaLLM: a local Ollama server (requires it running).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from config import Config

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. Cite sources inline as [n] matching the numbered context blocks. "
    "If the context does not contain the answer, say so plainly."
)


def build_prompt(question: str, contexts: List[str], history: str = "") -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    parts = [SYSTEM_PROMPT, ""]
    if history:
        parts += ["Conversation so far:", history, ""]
    parts += ["Context:", numbered, "", f"Question: {question}", "", "Answer:"]
    return "\n".join(parts)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: List[str], history: str = "") -> str:
        ...


class ExtractiveLLM(BaseLLM):
    """No-LLM fallback: return the most relevant passages with citations."""

    def __init__(self, max_passages: int = 3):
        self.max_passages = max_passages

    def generate(self, question: str, contexts: List[str], history: str = "") -> str:
        if not contexts:
            return "I couldn't find anything relevant in the indexed documents."
        lines = [
            "Based on the retrieved context (extractive mode — set an LLM "
            "provider for synthesized answers):",
            "",
        ]
        for i, ctx in enumerate(contexts[: self.max_passages], start=1):
            snippet = ctx.strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:500] + "…"
            lines.append(f"[{i}] {snippet}")
        return "\n".join(lines)


class GeminiLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, max_tokens: int, temperature: float):
        if not api_key:
            raise ValueError("Gemini LLM selected but no GEMINI_API_KEY is set.")
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, contexts: List[str], history: str = "") -> str:
        from google.genai import types

        prompt = build_prompt(question, contexts, history)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.max_tokens, temperature=self.temperature
            ),
        )
        return (resp.text or "").strip()


class OllamaLLM(BaseLLM):
    def __init__(self, model: str, host: str, temperature: float):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def generate(self, question: str, contexts: List[str], history: str = "") -> str:
        import requests

        prompt = build_prompt(question, contexts, history)
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


def get_llm(config: Config) -> BaseLLM:
    provider = config.llm_provider
    if provider == "gemini":
        return GeminiLLM(
            config.llm_model, config.gemini_api_key,
            config.max_tokens, config.temperature,
        )
    if provider == "ollama":
        return OllamaLLM(config.ollama_model, config.ollama_host, config.temperature)
    return ExtractiveLLM()
