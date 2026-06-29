"""Whisper STT — транскрибация в реальном времени."""

import time
import threading
import numpy as np
from typing import Optional, Callable
from faster_whisper import WhisperModel
from config import config

class Transcriber:
    """Whisper-based speech-to-text с VAD."""

    def __init__(self, on_text: Optional[Callable[[str, str], None]] = None):
        """
        on_text(text: str, source: str) — callback при распознанном тексте.
        source: "mic" или "system"
        """
        self.on_text = on_text
        self._model: Optional[WhisperModel] = None
        self._running = False
        self._queue: list[tuple[np.ndarray, str, float]] = []  # (audio, source, ts)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def load_model(self):
        """Загрузка Whisper модели."""
        if self._model is not None:
            return

        stt = config.stt
        print(f"Loading Whisper {stt.model_size} ({stt.device}/{stt.compute_type})...")
        self._model = WhisperModel(
            stt.model_size,
            device=stt.device,
            compute_type=stt.compute_type,
            cpu_threads=4,
            num_workers=1,
        )
        print("Whisper loaded ✓")

    def feed(self, audio: np.ndarray, source: str = "mic"):
        """Добавить аудио на обработку."""
        if len(audio) < config.audio.sample_rate * 0.5:  # минимум 0.5s
            return
        with self._lock:
            self._queue.append((audio.copy(), source, time.time()))

    def process_one(self, audio: np.ndarray, source: str) -> Optional[str]:
        """Транскрибировать один аудио-сегмент."""
        if self._model is None:
            return None
        segments, info = self._model.transcribe(
            audio,
            beam_size=config.stt.beam_size,
            vad_filter=config.stt.vad_filter,
            language=config.stt.language if config.stt.language != "auto" else None,
        )
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())
        return " ".join(texts) if texts else None

    def _loop(self):
        """Фоновый цикл транскрибации."""
        while self._running:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item:
                audio, source, ts = item
                text = self.process_one(audio, source)
                if text and self.on_text:
                    self.on_text(text, source)
            else:
                time.sleep(0.1)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)
