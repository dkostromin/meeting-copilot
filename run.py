"""Meeting Copilot — entry point."""

import argparse
import time
import threading
import sys
import re
import os
import glob

from config import config
from audio_capture import AudioCapture
from transcriber import Transcriber
from context import ContextRing
from llm_client import LlmClient


# Паттерны для детекта вопросов (проактивный режим)
QUESTION_PATTERNS = [
    r'\bкак\b.*\?',           # "как сделать?"
    r'\bчто\b.*\?',           # "что такое?"
    r'\bпочему\b.*\?',        # "почему не работает?"
    r'\bзачем\b.*\?',         # "зачем нужно?"
    r'\bкакой\b.*\?',         # "какой лучше?"
    r'\bсколько\b.*\?',       # "сколько стоит?"
    r'\bwhere\b.*\?',         # English
    r'\bhow\b.*\?',
    r'\bwhat\b.*\?',
    r'\bwhy\b.*\?',
    r'\bcan you\b.*\?',
    r'\bexplain\b',
    r'\bрасскажи\b',
    r'\bобъясни\b',
    r'\bпомоги\b',
]

# Слова-триrгеры для технических тем
TECH_TRIGGERS = [
    "error", "exception", "bug", "crash", "не работает", "ошибка",
    "api", "endpoint", "database", "sql", "query", "код", "функция",
    "deploy", "docker", "kubernetes", "config", "настройка",
]


def is_question(text: str) -> bool:
    """Детект: является ли фраза вопросом."""
    text_lower = text.lower().strip()
    if not text_lower or len(text_lower) < 5:
        return False
    # Прямой вопрос
    if text_lower.endswith("?"):
        return True
    # Паттерны
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def has_tech_trigger(text: str) -> bool:
    """Детект: есть ли техническая тема."""
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in TECH_TRIGGERS)


def _show_models_info():
    """Показать скачанные Whisper модели и путь к кэшу."""
    # Поиск кэша в стандартных местах
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub/models--*"),
        os.path.expanduser("~/.cache/faster_whisper/*"),
    ]

    print("📦 Whisper models cache:")
    found = False
    for pattern in cache_dirs:
        for p in sorted(glob.glob(pattern)):
            if os.path.isdir(p):
                size_mb = 0
                for root, dirs, files in os.walk(p):
                    for f in files:
                        try:
                            size_mb += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
                size_mb /= 1024 * 1024
                name = os.path.basename(p).replace("models--", "").replace("faster_whisper_", "")
                print(f"  📁 {name}: {size_mb:.0f} MB")
                print(f"     {p}")
                found = True

    if not found:
        cache_hf = os.path.expanduser("~/.cache/huggingface/hub/")
        cache_fw = os.path.expanduser("~/.cache/faster_whisper/")
        print(f"  (нет моделей в {cache_hf} или {cache_fw})")
        print(f"  Путь: {cache_hf}")
        print(f"  Путь: {cache_fw}")

    print()
    print("🗑  Удалить модель: rm -rf ~/.cache/huggingface/hub/models--*название*")
    print("   или: rm -rf ~/.cache/faster_whisper/*")
    print()

    # Рекомендация
    print("💡 Рекомендация: после тестов удалите ненужные модели:")
    print("   rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-*")
    print("   rm -rf ~/.cache/faster_whisper/*")
    print()


def main():
    parser = argparse.ArgumentParser(description="Meeting Copilot")
    parser.add_argument("--mock", action="store_true", help="Mock audio (dev)")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument("--no-ui", action="store_true", help="Терминальный режим без UI")
    parser.add_argument("--capture", action="store_true", help="Реальный захват аудио")
    parser.add_argument("--mic", action="store_true", help="Только микрофон (без BlackHole)")
    parser.add_argument("--proactive", action="store_true", help="Проактивный режим (автодетект вопросов)")
    parser.add_argument("--show-models", action="store_true", help="Показать скачанные модели и путь к кэшу")
    parser.add_argument("--model-path", action="store_true", help="Показать путь к кэшу моделей")
    args = parser.parse_args()

    # Режимы показа информации
    if args.show_models or args.model_path:
        _show_models_info()
        return 0

    config.mock_audio = args.mock or not args.capture
    config.debug = args.debug
    config.proactive = args.proactive
    config.mic_only = args.mic

    # Инициализация
    context = ContextRing()
    llm = LlmClient()

    # Проверка API ключа
    if not llm.api_key:
        print("⚠️  OPENCODE_GO_API_KEY не найден!")
        print("   export OPENCODE_GO_API_KEY='sk-...' или задайте в ~/.hermes/.env")
        return 1

    print("=" * 50)
    print("🎙  Meeting Copilot")
    print(f"🎤  Whisper: {config.stt.model_size} ({config.stt.device})")
    print(f"🤖  LLM: {config.llm.model} @ {config.llm.base_url}")
    print(f"📋  Контекст: {config.context.max_seconds // 60} мин")
    if config.proactive:
        print("🧠  Проактивный режим: ВКЛ")
    if config.mic_only:
        print("🎤  Только микрофон (без BlackHole)")
    if config.mock_audio:
        print("⚡  DEV MODE (mock audio)")
    print("=" * 50)

    # Callback для транскрибации
    def on_text(text: str, source: str):
        context.add(text, source)
        if config.debug:
            prefix = "🎤" if source == "mic" else "🔈"
            print(f"[{prefix}] {text}")

        # Проактивный режим — автодетект
        if config.proactive and is_question(text):
            if config.debug:
                print(f"\n🧠 Детект вопроса: {text[:80]}")
            _trigger_assistant(context, llm, overlay)

    # Транскрайбер
    transcriber = Transcriber(on_text=on_text)
    transcriber.load_model()
    transcriber.start()

    # Аудио-захват
    audio = AudioCapture()
    audio.start()

    # UI / Триггер
    overlay = None
    if not args.no_ui:
        from trigger import OverlayUI, HotkeyListener
        overlay = OverlayUI()
        threading.Thread(target=overlay.run, daemon=True).start()
        time.sleep(2)

        def on_trigger():
            _trigger_assistant(context, llm, overlay)

        hotkey = HotkeyListener(on_trigger=on_trigger)
        hotkey.start()
    else:
        print("\n📝 Терминальный режим. Enter = триггер, Ctrl+C = выход.\n")
        def _stdin_trigger():
            while True:
                try:
                    input()
                    _trigger_assistant(context, llm, overlay)
                except (EOFError, KeyboardInterrupt):
                    break
        threading.Thread(target=_stdin_trigger, daemon=True).start()

    # Main loop
    print("\n▶  Running. Press Ctrl+C to stop.\n")

    try:
        last_print = 0
        while True:
            chunk = audio.read(timeout=1.0)
            if chunk:
                transcriber.feed(chunk.data, chunk.source)

            if config.debug:
                now = time.time()
                if now - last_print > 30:
                    print(f"[Status] {context.count} фраз, {context.duration_seconds:.0f}s контекста, STT queue: {transcriber.pending}")
                    last_print = now

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n⏹  Stopping...")
    finally:
        audio.stop()
        transcriber.stop()
        if overlay:
            overlay.stop()
        print("Done.")
    return 0


def _trigger_assistant(context: ContextRing, llm: LlmClient, overlay):
    """Обработка триггера — запрос к LLM."""
    transcript = context.get_recent(15)
    if not transcript:
        msg = "Нет контекста для поиска"
        if overlay:
            overlay.show_result(msg)
        else:
            print(f"⚠️  {msg}")
        return

    if overlay:
        overlay.show_status("Анализирую...")
    else:
        print("🧠 Анализирую...")

    messages = context.get_llm_messages(15)
    response = llm.ask(
        user_message="Проанализируй контекст разговора и дай короткую подсказку. Если был задан вопрос — ответь на него.",
        context=messages,
    )

    context.add_assistant(response)

    if overlay:
        overlay.show_result(response)
    else:
        print(f"\n💡 {response}\n")

    if config.debug:
        print(f"[Ответ] {response}")


if __name__ == "__main__":
    sys.exit(main())
