"""Whisper STT backend — локальный faster-whisper."""

import time
import threading
import queue as _queue
from typing import Optional, Callable
import numpy as np

from config import config
from .base import SttBackend


class WhisperBackend(SttBackend):
    """faster-whisper на CPU/GPU с очередью чанков."""

    def __init__(self, on_text: Optional[Callable[[str, str], None]] = None):
        super().__init__(on_text)
        self._model = None
        self._queue: _queue.Queue = _queue.Queue(maxsize=50)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._load_state = "idle"  # idle | loading | ready | error

    # ── lifecycle ────────────────────────────────────────────

    def load_model(self, status_callback: Optional[Callable[[str], None]] = None):
        if self._model is not None:
            return

        stt = config.stt
        self._load_state = "loading"
        if status_callback:
            status_callback(f"Загрузка {stt.model_size}…")

        start_ts = time.time()

        # Спиннер в stdout
        stop_spinner = threading.Event()

        def _spinner():
            import sys
            chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            i = 0
            while not stop_spinner.is_set():
                elapsed = int(time.time() - start_ts)
                sys.stdout.write(f"\r  {chars[i]} Загрузка {stt.model_size}… {elapsed}s")
                sys.stdout.flush()
                i = (i + 1) % len(chars)
                if stop_spinner.wait(0.1):
                    break
            sys.stdout.write("\r" + " " * 40 + "\r")
            sys.stdout.flush()

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
            self._load_state = "ready"
        except Exception as e:
            self._load_state = "error"
            print(f"\n[Whisper] Ошибка загрузки: {e}")
            raise
        finally:
            stop_spinner.set()
            spinner_thread.join(timeout=1)

        total = int(time.time() - start_ts)
        print(f"✅ {stt.model_size} загружена ({total}s)")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ── data feeding ─────────────────────────────────────────

    def feed(self, audio: np.ndarray, source: str = "mic"):
        if len(audio) < config.audio.sample_rate * 0.5:
            return
        try:
            self._queue.put_nowait((audio.copy(), source, time.time()))
        except _queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((audio.copy(), source, time.time()))
            except _queue.Empty:
                pass

    def _loop(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except _queue.Empty:
                continue
            audio, source, ts = item
            text = self._process_one(audio, source)
            if text and self.on_text:
                try:
                    self.on_text(text, source)
                except Exception as e:
                    print(f"[Whisper] Ошибка callback: {e}")

    def _process_one(self, audio: np.ndarray, source: str) -> Optional[str]:
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
            print(f"[Whisper] Ошибка транскрибации: {e}")
            return None

    # ── status ───────────────────────────────────────────────

    def load_status(self) -> str:
        return self._load_state

    @property
    def pending(self) -> int:
        return self._queue.qsize()
