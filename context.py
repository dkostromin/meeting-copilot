"""Контекстное кольцо — rolling window последних фраз."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from config import config

@dataclass
class TranscriptEntry:
    text: str
    source: str  # "mic" | "system"
    timestamp: float
    role: str = "user"  # "user" | "assistant"

class ContextRing:
    """Rolling window транскриптов (5 мин по умолчанию)."""

    def __init__(self):
        self._entries: deque[TranscriptEntry] = deque(maxlen=config.context.max_pairs)
        self._llm_entries: list[dict] = []

    def add(self, text: str, source: str = "mic"):
        """Добавить распознанную фразу."""
        entry = TranscriptEntry(
            text=text,
            source=source,
            timestamp=__import__("time").time(),
            role="user",
        )
        self._entries.append(entry)

    def add_assistant(self, text: str):
        """Добавить ответ ассистента."""
        entry = TranscriptEntry(
            text=text,
            source="assistant",
            timestamp=__import__("time").time(),
            role="assistant",
        )
        self._entries.append(entry)

    def _trim_old(self):
        """Отсечь фразы старше max_seconds."""
        cutoff = __import__("time").time() - config.context.max_seconds
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    def get_recent(self, n: int = 10) -> str:
        """Последние N фраз как текст."""
        self._trim_old()
        entries = list(self._entries)[-n:]
        if not entries:
            return ""
        lines = []
        for e in entries:
            prefix = "🎤" if e.source == "mic" else "🔈"
            lines.append(f"{prefix} {e.text}")
        return "\n".join(lines)

    def get_llm_messages(self, n: int = 20) -> list[dict]:
        """Фразы для LLM (как сообщения)."""
        self._trim_old()
        entries = list(self._entries)[-n:]
        messages = []
        for e in entries:
            messages.append({"role": e.role, "content": e.text})
        return messages

    def get_transcript(self) -> str:
        """Полный транскрипт последних 5 минут."""
        self._trim_old()
        parts = []
        for e in self._entries:
            parts.append(f"[{__import__('datetime').datetime.fromtimestamp(e.timestamp).strftime('%H:%M:%S')}] {'🎤' if e.source == 'mic' else '🔈'} {e.text}")
        return "\n".join(parts)

    @property
    def duration_seconds(self) -> float:
        """Длительность контекста в секундах."""
        if not self._entries:
            return 0
        return self._entries[-1].timestamp - self._entries[0].timestamp
