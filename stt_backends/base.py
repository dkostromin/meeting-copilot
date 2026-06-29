"""Abstract base class for STT backends."""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np


class SttBackend(ABC):
    """Interface for all STT engines (Whisper, Deepgram, Google, etc.)."""

    def __init__(
        self,
        on_text: Optional[Callable[[str, str], None]] = None,
    ):
        """
        on_text(text: str, source: str) — callback при каждом распознанном тексте.
          source: "mic" | "system" (откуда пришло аудио)
        """
        self.on_text = on_text

    # ── lifecycle ────────────────────────────────────────────

    @abstractmethod
    def load_model(self, status_callback: Optional[Callable[[str], None]] = None):
        """One‑time initialisation (download/load model, open WS connection…).
        `status_callback` вызывается с короткими статусами для спиннера/UI."""
        ...

    @abstractmethod
    def start(self):
        """Запустить фоновый цикл обработки."""
        ...

    @abstractmethod
    def stop(self):
        """Остановить фоновый цикл, освободить ресурсы."""
        ...

    # ── data feeding ─────────────────────────────────────────

    @abstractmethod
    def feed(self, audio: np.ndarray, source: str = "mic"):
        """Добавить аудиочанк (16000 Hz, mono, float32) на обработку.
        source: "mic" или "system".
        """
        ...

    # ── status ───────────────────────────────────────────────

    @abstractmethod
    def load_status(self) -> str:
        """Статус загрузки: 'loading' | 'ready' | 'error'."""
        ...

    @property
    def pending(self) -> int:
        """Количество необработанных элементов очереди (0 для стриминговых бекендов)."""
        return 0
