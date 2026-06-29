# Meeting Copilot 🎙️

Proactive AI assistant for meetings. Listens to your calls, understands context, and shows relevant insights — all locally on your Mac.

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Download Whisper model
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='metal', compute_type='float16')"

# 3. Set API key
export OPENCODE_GO_API_KEY="sk-..."

# 4. Run
python run.py --capture        # Микрофон + системный звук (с BlackHole)
python run.py --capture --mic  # Только микрофон
python run.py --mock           # Дев-режим (без аудио)

# Для проактивного режима (сам подсказывает без хоткея):
python run.py --capture --proactive

# Без UI (терминальный лог):
python run.py --capture --no-ui
```

## Features

- **Realtime STT** — faster-whisper large-v3, Metal acceleration на Apple Silicon
- **Proactive mode** — AI сам детектит вопросы в разговоре и даёт подсказки
- **Manual mode** — `Cmd+Shift+H` → ищет контекст и отвечает в оверлее
- **Rolling context** — последние 5 минут разговора
- **Overlay** — плавающее окно PyQt6 (невидимое при скриншарях)
- **LLM** — любой OpenAI-compatible API (OpenCode, DeepSeek, Ollama, etc.)

## Audio Setup

### С BlackHole (системный звук + микрофон):
```bash
brew install blackhole-2ch
```
Открой Audio MIDI Setup → создай Multi-Output Device (BlackHole + наушники).
В Zoom/Teams поставь BlackHole как динамик.

### Только микрофон:
Просто запусти `python run.py --capture --mic`

## Config

Смотри `config.py` — всё через dataclasses:
- `LlmConfig` — model, base_url, api_key
- `SttConfig` — model_size (large-v3 / distil-large-v3 / medium), device (auto/metal/cpu)
- `OverlayConfig` — hotkey, position, size

## Project Structure

```
meeting-copilot/
├── run.py              # Entry point
├── config.py           # Dataclass config
├── audio_capture.py    # sounddevice: mic + BlackHole → 16kHz mono
├── transcriber.py      # faster-whisper + VAD
├── context.py          # Rolling window 5 min
├── llm_client.py       # OpenAI-compatible API
├── trigger.py          # PyQt6 overlay + hotkey
├── requirements.txt
└── README.md
```
