"""Конфигурация Meeting Copilot."""

import os
import platform
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """Настройки захвата аудио."""
    device_mic: Optional[str] = None           # Микрофон (None = default)
    device_system: Optional[str] = None        # BlackHole (None = default)
    sample_rate: int = 16000                   # Для Whisper
    chunk_seconds: float = 2.0                 # Размер чанка для STT


@dataclass
class SttConfig:
    """Whisper STT."""
    model_size: str = "medium"                 # medium — оптимально для быстродействия на Mac CPU
    device: str = "auto"                       # auto / metal / cpu
    compute_type: str = "auto"                 # auto / float16 / int8
    language: str = "auto"                     # auto / ru / en
    beam_size: int = 5
    vad_filter: bool = True

    def __post_init__(self):
        if self.device == "auto":
            # faster-whisper на Mac не поддерживает device="metal"
            # CTranslate2 сам использует ARM NEON оптимизации на CPU
            self.device = "cpu"
        if self.compute_type == "auto":
            # int8 — оптимально для CPU/ARM NEON
            self.compute_type = "int8"


@dataclass
class LlmConfig:
    """OpenAI-compatible API."""
    api_key: str = ""
    base_url: str = "https://opencode.ai/zen/go/v1"
    model: str = "deepseek-v4-flash"
    max_tokens: int = 512
    temperature: float = 0.3
    timeout: int = 60
    system_prompt: str = (
        "Ты — AI-ассистент на созвоне. "
        "Отвечай кратко и по делу. "
        "Если информации не хватает — скажи. "
        "Пиши на том же языке, на котором идёт разговор."
    )

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("OPENCODE_GO_API_KEY", "")
            if not self.api_key:
                env_path = os.path.expanduser("~/.hermes/.env")
                if os.path.exists(env_path):
                    with open(env_path) as f:
                        for line in f:
                            if "OPENCODE_GO_API_KEY" in line:
                                self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                break

        # Переопределение через env
        env_base = os.environ.get("LLM_BASE_URL", "")
        if env_base:
            self.base_url = env_base
        env_model = os.environ.get("LLM_MODEL", "")
        if env_model:
            self.model = env_model


@dataclass
class ContextConfig:
    """Контекстное окно."""
    max_seconds: int = 300                     # 5 минут контекста
    max_pairs: int = 50                        # макс. фраз


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
    proactive: bool = False                    # автодетект вопросов
    mic_only: bool = False                     # только микрофон (без BlackHole)


config = Config()
