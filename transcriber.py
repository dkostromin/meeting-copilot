"""Transcriber — фасад над STT-бекендом (Whisper, Deepgram, Google)."""

from typing import Optional, Callable
from stt_backends import create_backend
from stt_backends.base import SttBackend


class Transcriber:
    """Единый интерфейс для любого STT-бекенда.

    Использование:
        t = Transcriber(on_text=lambda text, source: ...)
        t.load_model()
        t.start()
        t.feed(audio_chunk, source="mic")
        ...
        t.stop()
    """

    def __init__(self, on_text: Optional[Callable[[str, str], None]] = None):
        self.on_text = on_text
        self._backend: Optional[SttBackend] = None

    @property
    def backend(self) -> Optional[SttBackend]:
        return self._backend

    def load_model(self):
        """Создать и загрузить бекенд согласно config.stt.backend."""
        self._backend = create_backend(on_text=self.on_text)
        self._backend.load_model()

    def start(self):
        assert self._backend is not None, "Загрузки не было — вызови load_model()"
        self._backend.start()

    def stop(self):
        if self._backend:
            self._backend.stop()

    def feed(self, audio, source: str = "mic"):
        if self._backend:
            self._backend.feed(audio, source)

    @property
    def pending(self) -> int:
        """Количество необработанных чанков (только для Whisper)."""
        return self._backend.pending if self._backend else 0

    def load_status(self) -> str:
        return self._backend.load_status() if self._backend else "idle"
