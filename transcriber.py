"""Whisper STT — транскрибация в реальном времени."""

import time
import threading
import queue as _queue
import numpy as np
from typing import Optional, Callable
from config import config


class Transcriber:
    """Whisper-based speech-to-text с VAD."""

    def __init__(self, on_text: Optional[Callable[[str, str], None]] = None):
        """
        on_text(text: str, source: str) — callback при распознанном тексте.
        source: "mic" или "system"
        """
        self.on_text = on_text
        self._model = None
        self._running = False
        self._queue: _queue.Queue = _queue.Queue(maxsize=50)
        self._thread: Optional[threading.Thread] = None

    def load_model(self):
        """Загрузка Whisper модели с прогресс-баром."""
        if self._model is not None:
            return

        stt = config.stt
        print(f"Loading Whisper {stt.model_size} ({stt.device}/{stt.compute_type})...\n")

        # Спиннер в отдельном потоке
        stop_spinner = threading.Event()

        def _spinner():
            chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            i = 0
            while not stop_spinner.is_set():
                sys.stdout.write(f"\r{' ' * 40}\rDownloading model {chars[i]}")
                sys.stdout.flush()
                i = (i + 1) % len(chars)
                if stop_spinner.wait(0.08):
                    break
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

        import sys
        spinner_thread = threading.Thread(target=_spinner, daemon=True)
        spinner_thread.start()

        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                stt.model_size,
                device=stt.device,
                compute_type=stt.compute_type,
                cpu_threads=4,
                num_workers=1,
            )
        finally:
            stop_spinner.set()
            spinner_thread.join(timeout=1)

        print("✅ Whisper loaded ✓")

    def feed(self, audio: np.ndarray, source: str = "mic"):
        """Добавить аудио на обработку."""
        if len(audio) < config.audio.sample_rate * 0.5:
            return
        try:
            self._queue.put_nowait((audio.copy(), source, time.time()))
        except _queue.Full:
            # Дропаем самый старый чанк
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((audio.copy(), source, time.time()))
            except _queue.Empty:
                pass

    def process_one(self, audio: np.ndarray, source: str) -> Optional[str]:
        """Транскрибировать один аудио-сегмент."""
        if self._model is None:
            return None
        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=config.stt.beam_size,
                vad_filter=config.stt.vad_filter,
                language=config.stt.language if config.stt.language != "auto" else None,
            )
            texts = []
            for seg in segments:
                t = seg.text.strip()
                if t:
                    texts.append(t)
            return " ".join(texts) if texts else None
        except Exception as e:
            print(f"[STT] Ошибка транскрибации: {e}")
            return None

    def _loop(self):
        """Фоновый цикл транскрибации."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except _queue.Empty:
                continue

            audio, source, ts = item
            text = self.process_one(audio, source)
            if text and self.on_text:
                try:
                    self.on_text(text, source)
                except Exception as e:
                    print(f"[STT] Ошибка callback: {e}")

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
        return self._queue.qsize()
