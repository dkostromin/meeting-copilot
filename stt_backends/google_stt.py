"""Google Cloud STT backend — streaming recognition через gRPC."""

import os
import time
import queue as _queue
import threading
from typing import Optional, Callable
import numpy as np

from config import config
from .base import SttBackend


class GoogleSttBackend(SttBackend):
    """Google Cloud Speech-to-Text (streaming).
    Использует standard модель (Chirp пока не для стриминга русского).
    """

    def __init__(self, on_text: Optional[Callable[[str, str], None]] = None):
        super().__init__(on_text)
        self._client = None
        self._queue: _queue.Queue = _queue.Queue(maxsize=50)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._load_state = "idle"

    # ── lifecycle ────────────────────────────────────────────

    def load_model(self, status_callback: Optional[Callable[[str], None]] = None):
        """Проверяем credentials, создаём клиент."""
        creds_path = config.stt.google_credentials_path
        if not creds_path:
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

        if not creds_path or not os.path.exists(creds_path):
            self._load_state = "error"
            msg = (
                "Google credentials не найдены! Укажи путь в GOOGLE_APPLICATION_CREDENTIALS\n"
                "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
            )
            print(f"[Google STT] ⚠️ {msg}")
            raise RuntimeError(msg)

        try:
            from google.cloud.speech_v2 import SpeechClient
            from google.api_core.client_options import ClientOptions

            self._client = SpeechClient(
                client_options=ClientOptions(
                    api_endpoint="speech.googleapis.com"
                )
            )
            self._load_state = "ready"
            print("[Google STT] ✅ Клиент создан")
        except ImportError:
            self._load_state = "error"
            msg = "google-cloud-speech не установлен: pip install google-cloud-speech"
            print(f"[Google STT] ⚠️ {msg}")
            raise RuntimeError(msg)
        except Exception as e:
            self._load_state = "error"
            print(f"[Google STT] Ошибка инициализации: {e}")
            raise

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
        """Фоновый цикл — последовательные streamingRecognize запросы."""
        if self._client is None:
            print("[Google STT] Клиент не инициализирован")
            return

        lang = config.stt.language if config.stt.language != "auto" else "ru-RU"
        # Определяем project из credentials
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project_id:
            # Пробуем прочитать из JSON ключа
            import json as _json
            try:
                creds = config.stt.google_credentials_path
                if creds and os.path.exists(creds):
                    with open(creds) as f:
                        data = _json.load(f)
                        project_id = data.get("project_id", "")
            except Exception:
                pass

        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except _queue.Empty:
                continue

            audio, source, ts = item
            text = self._transcribe(audio, lang, project_id, source)
            if text and self.on_text:
                try:
                    self.on_text(text, source)
                except Exception as e:
                    print(f"[Google STT] Ошибка callback: {e}")

    def _transcribe(
        self,
        audio: np.ndarray,
        language: str,
        project_id: str,
        source: str,
    ) -> Optional[str]:
        if self._client is None:
            return None
        try:
            from google.cloud.speech_v2 import RecognitionConfig, AutoDetectDecodingConfig, ExplicitDecodingConfig
            from google.cloud.speech_v2.types import cloud_speech

            # Конфиг
            config_obj = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=[language],
                model="latest_short",
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False,
                ),
            )

            # Аудио контент
            content = (audio * 32767).astype(np.int16).tobytes()

            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{project_id}/locations/global/recognizers/_",
                config=config_obj,
                content=content,
            )

            response = self._client.recognize(request=request)

            texts = []
            for result in response.results:
                if result.alternatives:
                    t = result.alternatives[0].transcript.strip()
                    if t:
                        texts.append(t)
            return " ".join(texts) if texts else None

        except Exception as e:
            print(f"[Google STT] Ошибка: {e}")
            return None

    # ── status ───────────────────────────────────────────────

    def load_status(self) -> str:
        return self._load_state

    @property
    def pending(self) -> int:
        return self._queue.qsize()
