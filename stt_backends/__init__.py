"""STT backend selection factory."""

from typing import Optional, Callable
from config import config


def create_backend(on_text: Optional[Callable[[str, str], None]] = None):
    """Создать STT-бекенд согласно config.stt.backend."""
    backend_name = config.stt.backend

    if backend_name == "whisper":
        from .whisper_stt import WhisperBackend
        print(f"🎤 STT бекенд: Whisper ({config.stt.model_size})")
        return WhisperBackend(on_text=on_text)

    elif backend_name == "deepgram":
        from .deepgram_stt import DeepgramBackend
        print("🎤 STT бекенд: Deepgram Nova-3 (real-time streaming)")
        return DeepgramBackend(on_text=on_text)

    elif backend_name == "google":
        from .google_stt import GoogleSttBackend
        print("🎤 STT бекенд: Google Cloud Speech-to-Text")
        return GoogleSttBackend(on_text=on_text)

    else:
        raise ValueError(f"Неизвестный STT бекенд: {backend_name!r}. "
                         f"Доступны: whisper, deepgram, google")
