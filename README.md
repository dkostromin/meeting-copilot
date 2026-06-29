# Meeting Copilot 🎙️

Proactive AI assistant for meetings. Listens to your calls, detects questions, and shows relevant answers — all locally on your Mac.

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. System deps (macOS)
brew install portaudio   # for sounddevice

# 3. Set API key
export OPENCODE_GO_API_KEY="sk-..."

# 4. Run (только микрофон, без оверлея — для теста)
python run.py --capture --mic --no-ui --debug

# 5. Run (с оверлеем)
python run.py --capture --mic

# 6. Run (проактивный — сам детектит вопросы)
python run.py --capture --mic --proactive

# 7. Run (с системным звуком — нужен BlackHole)
python run.py --capture

# 8. Dev-режим (без аудио)
python run.py --mock --no-ui --debug
```

## Flags

| Flag | Что делает |
|------|------------|
| `--capture` | Реальный захват аудио (иначе mock) |
| `--mic` | Только микрофон (без BlackHole) |
| `--proactive` | Автодетект вопросов → AI отвечает сам |
| `--no-ui` | Терминальный режим (без PyQt6 оверлея) |
| `--debug` | Подробный вывод |
| `--mock` | Фейковое аудио (для разработки) |

## Audio Setup (опционально)

### С BlackHole (системный звук + микрофон):
```bash
brew install blackhole-2ch
```
Открой Audio MIDI Setup → создай Multi-Output Device (BlackHole + наушники).
В Zoom/Teams поставь BlackHole как динамик.

### Только микрофон:
Просто запусти `python run.py --capture --mic`

## Config

Смотри `config.py` — всё через dataclasses.
Переопределение через env: `LLM_BASE_URL`, `LLM_MODEL`, `OPENCODE_GO_API_KEY`.

## Project Structure

```
meeting-copilot/
├── run.py              # Entry point, main loop, проактивный детект
├── config.py           # Dataclass config (LLM, Audio, STT, Overlay)
├── audio_capture.py    # sounddevice: mic + BlackHole → 16kHz mono
├── transcriber.py      # faster-whisper + VAD, bounded queue
├── context.py          # Rolling window 5 мин
├── llm_client.py       # OpenAI-compatible API (stream + sync)
├── trigger.py          # PyQt6 overlay + pynput хоткей
├── requirements.txt
└── README.md
```
