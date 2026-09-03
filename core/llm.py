"""Answer generation behind a swappable interface.

Providers:
  * ``none``   -> ExtractiveLLM: no API key needed; stitches the top passages
                  into an answer with inline [n] citations. Works fully offline.
  * ``gemini`` -> GeminiLLM: Google Gemini (requires GEMINI_API_KEY).
  * ``anthropic`` -> AnthropicLLM: Claude via the official Anthropic SDK
    (requires ANTHROPIC_API_KEY).
  * ``ollama`` -> OllamaLLM: a local Ollama server (requires it running).
  * OpenAI-compatible -> OpenAICompatibleLLM: any endpoint speaking the OpenAI
    Chat Completions API. Presets: ``openai``, ``nvidia`` (build.nvidia.com),
    ``groq``, ``together``, ``openrouter``; or ``openai`` + OPENAI_BASE_URL for
    a local vLLM / LM Studio / any other compatible server.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

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


def build_messages(
    question: str, contexts: List[str], history: str = ""
) -> List[Dict[str, str]]:
    """Chat-format variant: system prompt + a single user turn with context.

    Used by chat-completion providers (OpenAI, NVIDIA, ...) where separating the
    system role from the user content yields better instruction-following.
    """
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    user_parts = []
    if history:
        user_parts += ["Conversation so far:", history, ""]
    user_parts += ["Context:", numbered, "", f"Question: {question}"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


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


class AnthropicLLM(BaseLLM):
    """Claude via the official Anthropic SDK (Messages API).

    Note: no ``temperature`` is sent — Claude Opus 4.8/4.7 reject sampling
    parameters with a 400. Steer behaviour through the prompt instead.
    """

    def __init__(self, model: str, api_key: str, max_tokens: int):
        if not api_key:
            raise ValueError(
                "Anthropic LLM selected but no ANTHROPIC_API_KEY is set."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic "
                "provider. Install it with: pip install anthropic"
            ) from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, contexts: List[str], history: str = "") -> str:
        messages = build_messages(question, contexts, history)
        # Anthropic takes the system prompt as a top-level argument, so split it
        # out of the message list.
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT
        )
        chat = [m for m in messages if m["role"] != "system"]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat,
        )
        return "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()


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


class OpenAICompatibleLLM(BaseLLM):
    """Any OpenAI Chat Completions-compatible endpoint.

    One class covers OpenAI, NVIDIA (build.nvidia.com), Groq, Together,
    OpenRouter, and local servers (vLLM, LM Studio, Ollama's OpenAI shim) —
    they differ only by ``base_url`` and ``api_key``.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        provider: str = "openai",
    ):
        if not api_key:
            raise ValueError(
                f"LLM provider '{provider}' selected but no API key is set. "
                "Set OPENAI_API_KEY (or the provider's own key env var, e.g. "
                "NVIDIA_API_KEY)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAI-compatible "
                "providers. Install it with: pip install openai"
            ) from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, contexts: List[str], history: str = "") -> str:
        messages = build_messages(question, contexts, history)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm(config: Config) -> BaseLLM:
    provider = config.llm_provider
    if provider == "gemini":
        return GeminiLLM(
            config.llm_model, config.gemini_api_key,
            config.max_tokens, config.temperature,
        )
    if provider == "anthropic":
        return AnthropicLLM(
            config.anthropic_model, config.anthropic_api_key, config.max_tokens,
        )
    if provider == "ollama":
        return OllamaLLM(config.ollama_model, config.ollama_host, config.temperature)
    # Any OpenAI-compatible provider (openai, nvidia, groq, together, openrouter,
    # or a custom endpoint via OPENAI_BASE_URL).
    if provider in {"openai", "nvidia", "groq", "together", "openrouter"}:
        return OpenAICompatibleLLM(
            config.llm_model,
            config.openai_api_key,
            config.resolve_openai_base_url(),
            config.max_tokens,
            config.temperature,
            provider=provider,
        )
    return ExtractiveLLM()
