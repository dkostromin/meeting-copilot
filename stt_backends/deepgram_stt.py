"""Deepgram STT backend — real-time streaming через WebSocket."""

import os
import sys
import json
import asyncio
import threading
import time
from typing import Optional, Callable
import numpy as np

from config import config
from .base import SttBackend


class DeepgramBackend(SttBackend):
    """Deepgram Nova-3 через WebSocket API.
    Real-time streaming, latency ~0.5-1.5s, без очереди.
    """

    def __init__(self, on_text: Optional[Callable[[str, str], None]] = None):
        super().__init__(on_text)
        self._api_key = config.stt.deepgram_api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not self._api_key:
            print("[Deepgram] Ищем ключ в DEEPGRAM_API_KEY...")

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._load_state = "idle"

        # Очередь аудио для отправки в WS (из любого потока → asyncio)
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    # ── lifecycle ────────────────────────────────────────────

    def load_model(self, status_callback: Optional[Callable[[str], None]] = None):
        """Проверка API ключа — никаких моделей скачивать не нужно."""
        if self._api_key:
            self._load_state = "ready"
        else:
            self._load_state = "error"
            msg = "Deepgram API key не найден! Укажи DEEPGRAM_API_KEY."
            print(f"[Deepgram] ⚠️ {msg}")
            raise RuntimeError(msg)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        # Даём время на коннект
        time.sleep(1.5)

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)

    def _run_event_loop(self):
        """Фоновый asyncio event loop для WS."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_loop())
        except Exception as e:
            print(f"[Deepgram] Event loop error: {e}")

    async def _ws_loop(self):
        """Подключение, отправка аудио, приём результатов."""
        lang = config.stt.language if config.stt.language != "auto" else "ru"
        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?encoding=linear16"
            f"&sample_rate=16000"
            f"&channels=1"
            f"&language={lang}"
            f"&model=nova-3"
            f"&smart_format=true"
            f"&interim_results=false"
            f"&endpointing=300"           # автоматическая пауза через 300мс тишины
            f"&utterance_end_ms=1000"      # финализация фразы через 1с тишины
        )

        headers = {
            "Authorization": f"Token {self._api_key}",
        }

        try:
            import aiohttp
        except ImportError:
            print("[Deepgram] Ошибка: установи aiohttp — pip install aiohttp")
            return

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    async with session.ws_connect(url, headers=headers) as ws:
                        self._ws = ws
                        print("[Deepgram] ✅ Подключено к Deepgram")

                        # Фоновый таск: отправка аудио
                        async def send_audio():
                            while self._running:
                                chunk, source = await self._audio_queue.get()
                                # float32 0..1 → int16 для Deepgram
                                int16 = (chunk * 32767).astype(np.int16).tobytes()
                                await ws.send_bytes(int16)

                        send_task = asyncio.create_task(send_audio())

                        # Keepalive каждые 5с
                        async def keepalive():
                            while self._running:
                                await asyncio.sleep(5)
                                try:
                                    await ws.send_str(json.dumps({"type": "KeepAlive"}))
                                except Exception:
                                    break

                        ka_task = asyncio.create_task(keepalive())

                        # Приём результатов
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                self._handle_msg(data, msg=msg)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(f"[Deepgram] WS ошибка: {msg.data}")
                                break

                        send_task.cancel()
                        ka_task.cancel()
                        self._ws = None
                        print("[Deepgram] 🔌 Отключились, переподключаюсь...")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[Deepgram] Ошибка коннекта: {e}")
                    if self._running:
                        await asyncio.sleep(3)  # retry

    def _handle_msg(self, data: dict, msg=None):
        """Обработать входящее сообщение Deepgram."""
        if data.get("type") == "Results":
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if alternatives:
                alt = alternatives[0]
                transcript = alt.get("transcript", "").strip()
                is_final = data.get("is_final", False)

                if transcript and is_final and self.on_text:
                    # source всегда "mic" для Deepgram (микрофон + системный микс)
                    try:
                        self.on_text(transcript, "mic")
                    except Exception as e:
                        print(f"[Deepgram] Ошибка callback: {e}")

    # ── data feeding ─────────────────────────────────────────

    def feed(self, audio: np.ndarray, source: str = "mic"):
        """Отправить аудио в Deepgram WS (неблокирующая)."""
        if self._loop is None or not self._loop.is_running():
            return
        try:
            self._audio_queue.put_nowait((audio, source))
        except asyncio.QueueFull:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait((audio, source))
            except asyncio.QueueEmpty:
                pass

    # ── status ───────────────────────────────────────────────

    def load_status(self) -> str:
        return self._load_state

    @property
    def pending(self) -> int:
        """Deepgram — реальный стриминг, очереди нет."""
        return 0
