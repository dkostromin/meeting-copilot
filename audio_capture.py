"""Захват аудио: микрофон + системный звук (BlackHole)."""

import time
import queue
import threading
import numpy as np
from typing import Optional
from dataclasses import dataclass
from config import config


@dataclass
class AudioChunk:
    """Чанк аудио с метаданными."""
    data: np.ndarray                          # float32, 16000 Hz, mono
    timestamp: float                          # time.time()
    source: str = "mic"                       # "mic" | "system"


class AudioCapture:
    """Захват с микрофона и BlackHole, микс в моно 16kHz."""

    def __init__(self):
        self.buffer: queue.Queue[AudioChunk] = queue.Queue(maxsize=100)
        self._running = False
        self._threads: list[threading.Thread] = []
        self._sr = config.audio.sample_rate

    def _find_device(self, name_hint: str) -> Optional[int]:
        """Найти устройство по имени, вернуть индекс."""
        import sounddevice as sd
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) == 0:
                    continue
                if name_hint and name_hint.lower() in d["name"].lower():
                    return i
        except Exception:
            pass
        return None

    def _capture_loop(self, device_idx: Optional[int], source: str):
        """Цикл захвата с одного устройства."""
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                return
            # Downmix to mono
            data = indata.mean(axis=1) if indata.ndim == 2 and indata.shape[1] > 1 else indata.flatten()
            chunk = AudioChunk(
                data=data.astype(np.float32),
                timestamp=time.time(),
                source=source,
            )
            try:
                self.buffer.put_nowait(chunk)
            except queue.Full:
                # Дропаем старые чанки, если буфер переполнен
                try:
                    self.buffer.get_nowait()
                    self.buffer.put_nowait(chunk)
                except queue.Empty:
                    pass

        try:
            stream = sd.InputStream(
                device=device_idx,
                samplerate=self._sr,
                channels=1,
                callback=callback,
                blocksize=int(self._sr * config.audio.chunk_seconds),
            )
            with stream:
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[Audio] Ошибка захвата ({source}): {e}")

    def _mock_loop(self):
        """Mock-генератор для dev-тестов (тишина)."""
        sr = self._sr
        while self._running:
            chunk_size = int(sr * config.audio.chunk_seconds)
            data = np.random.randn(chunk_size).astype(np.float32) * 0.001
            try:
                self.buffer.put(AudioChunk(data=data, timestamp=time.time(), source="mic"), timeout=2)
            except queue.Full:
                pass
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

        # BlackHole (системный звук), если не mic_only и найден
        if not config.mic_only:
            bh_idx = self._find_device("blackhole")
            if bh_idx is not None:
                t = threading.Thread(target=self._capture_loop, args=(bh_idx, "system"), daemon=True)
                t.start()
                self._threads.append(t)
            elif config.debug:
                print("[Audio] BlackHole не найден — только микрофон")

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
