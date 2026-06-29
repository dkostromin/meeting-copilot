"""Конфигурация Meeting Copilot MVP."""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AudioConfig:
    """Настройки захвата аудио."""
    device_mic: Optional[str] = None           # Микрофон (None = default)
    device_system: Optional[str] = None        # BlackHole (None = default)
    sample_rate: int = 16000                   # Для Whisper
    chunk_seconds: float = 2.0                 # Размер чанка для STT
    vad_threshold: float = 0.5                 # Voice Activity Detection

@dataclass
class SttConfig:
    """Whisper STT."""
    model_size: str = "large-v3"              # tiny/base/small/medium/large-v3
    device: str = "auto"                       # auto / metal / cpu
    compute_type: str = "auto"                 # auto / float16 / int8
    language: str = "auto"                     # auto / ru / en
    beam_size: int = 5
    vad_filter: bool = True

    def __post_init__(self):
        if self.device == "auto":
            import platform
            self.device = "metal" if platform.system() == "Darwin" else "cpu"
        if self.compute_type == "auto":
            self.compute_type = "float16" if self.device == "metal" else "int8"

@dataclass
class LlmConfig:
    """OpenCode / OpenAI-compatible API."""
    api_key: str = ""
    base_url: str = ""                              # По умолч. OpenCode API, можно кастомный
    model: str = "deepseek-v4-flash"
    max_tokens: int = 512
    temperature: float = 0.3
    system_prompt: str = (
        "Ты — AI-ассистент для созвонов по NetSuite. "
        "Твоя задача — помогать быстро отвечать на вопросы клиентов. "
        "Отвечай кратко и по делу. Если информации не хватает — скажи."
    )

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("OPENCODE_GO_API_KEY", "")
            if not self.api_key:
                # Пробуем .env
                env_path = os.path.expanduser("~/.hermes/.env")
                if os.path.exists(env_path):
                    with open(env_path) as f:
                        for line in f:
                            if "OPENCODE_GO_API_KEY" in line:
                                self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                break

@dataclass
class ContextConfig:
    """Контекстное окно."""
    max_seconds: int = 300                     # 5 минут контекста
    max_pairs: int = 50                         # макс. фраз

@dataclass
class OverlayConfig:
    """Плавающее окно."""
    hotkey: str = "cmd+shift+h"
    opacity: float = 0.92
    width: int = 420
    height: int = 280
    font_size: int = 13
    position: str = "bottom-right"             # bottom-right, top-right, center

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    mock_audio: bool = False
    debug: bool = False

    def setup(self):
        """Auto-detect missing config values."""
        if self.stt.device == "auto":
            import platform
            self.stt.device = "metal" if platform.system() == "Darwin" else "cpu"
            self.stt.compute_type = "float16" if self.stt.device == "metal" else "int8"

config = Config()
