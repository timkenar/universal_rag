"""Sliding-window conversation memory.

Keeps the last ``memory_window`` turns so multi-turn ``chat`` sessions can
resolve follow-up references ("what about the second one?") in the LLM prompt.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List


@dataclass
class Turn:
    user: str
    assistant: str


class ConversationMemory:
    def __init__(self, window: int = 5):
        self.window = window
        self._turns: deque[Turn] = deque(maxlen=window)

    def add(self, user: str, assistant: str) -> None:
        self._turns.append(Turn(user=user, assistant=assistant))

    def turns(self) -> List[Turn]:
        return list(self._turns)

    def format(self) -> str:
        """Render history as a plain-text transcript for the prompt."""
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()
