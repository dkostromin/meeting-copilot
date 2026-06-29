"""Захват аудио: микрофон + системный звук (BlackHole)."""

import time
import queue
import threading
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field
from config import config

@dataclass
class AudioChunk:
    """Чанк аудио с метаданными."""
    data: np.ndarray                          # float32, 16000 Hz, mono
    timestamp: float                          # time.time()
    source: str = "mic"                       # "mic" | "system" | "mixed"

class AudioCapture:
    """Захват с микрофона и BlackHole, микс в моно 16kHz."""

    def __init__(self):
        self.buffer: queue.Queue[AudioChunk] = queue.Queue()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._sr = config.audio.sample_rate

    def _find_device(self, name_hint: str, kind: str = "input"):
        """Найти устройство по имени."""
        import sounddevice as sd
        for dev in sd.query_devices():
            if kind == "input" and dev.get("max_input_channels", 0) == 0:
                continue
            if kind == "output" and dev.get("max_output_channels", 0) == 0:
                continue
            if name_hint and name_hint.lower() in dev["name"].lower():
                return dev["index"]
        return None

    def _capture_loop(self, device: Optional[str], source: str):
        """Цикл захвата с одного устройства."""
        import sounddevice as sd
        idx = None
        if device:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if device.lower() in d["name"].lower():
                    idx = i
                    break

        def callback(indata, frames, time_info, status):
            if status:
                return
            # Downmix to mono if needed
            data = indata.mean(axis=1) if indata.ndim == 2 and indata.shape[1] > 1 else indata.flatten()
            # Resample if needed (assume 48kHz from system → 16kHz)
            if self._sr != int(sd.query_devices(idx)["default_samplerate"]) if idx else 48000:
                data = self._resample(data, 48000 if source == "system" else 48000, self._sr)
            chunk = AudioChunk(data=data.astype(np.float32), timestamp=time.time(), source=source)
            self.buffer.put(chunk)

        stream = sd.InputStream(
            device=idx,
            samplerate=self._sr,
            channels=1,
            callback=callback,
            blocksize=int(self._sr * config.audio.chunk_seconds),
        )
        with stream:
            while self._running:
                time.sleep(0.1)

    def _resample(self, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Простой resample (линейная интерполяция)."""
        if orig_sr == target_sr:
            return data
        ratio = target_sr / orig_sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        return np.interp(indices, np.arange(len(data)), data)

    def _mock_loop(self):
        """Mock-генератор для dev-тестов (тишина)."""
        sr = self._sr
        while self._running:
            chunk_size = int(sr * config.audio.chunk_seconds)
            data = np.random.randn(chunk_size).astype(np.float32) * 0.001  # тишина
            self.buffer.put(AudioChunk(data=data, timestamp=time.time(), source="mic"))
            time.sleep(config.audio.chunk_seconds)

    def start(self):
        """Запуск захвата."""
        self._running = True

        if config.mock_audio:
            t = threading.Thread(target=self._mock_loop, daemon=True)
            t.start()
            self._threads.append(t)
            return

        # Микрофон (по умолчанию)
        t = threading.Thread(target=self._capture_loop, args=(None, "mic"), daemon=True)
        t.start()
        self._threads.append(t)

        # BlackHole (системный звук), если найден
        bh_idx = self._find_device("blackhole")
        if bh_idx is not None:
            t = threading.Thread(target=self._capture_loop, args=("blackhole", "system"), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2)

    def read(self, timeout: float = 2.0) -> Optional[AudioChunk]:
        """Прочитать следующий чанк."""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_window(self, seconds: float = 5.0) -> np.ndarray:
        """Собрать окно аудио за N секунд из буфера."""
        chunks = []
        cutoff = time.time() - seconds
        # Чтобы не блокировать — берём что есть
        while not self.buffer.empty():
            try:
                chunk = self.buffer.get_nowait()
                if chunk.timestamp >= cutoff:
                    chunks.append(chunk.data)
            except queue.Empty:
                break
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)
