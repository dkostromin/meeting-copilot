#!/usr/bin/env python3
"""Meeting Copilot — MVP entry point."""

import argparse
import time
import threading
import sys

from config import config
from audio_capture import AudioCapture
from transcriber import Transcriber
from context import ContextRing
from llm_client import LlmClient

def main():
    parser = argparse.ArgumentParser(description="Meeting Copilot MVP")
    parser.add_argument("--mock", action="store_true", help="Mock audio (dev)")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument("--no-ui", action="store_true", help="Терминальный режим без UI")
    parser.add_argument("--capture", action="store_true", help="Реальный захват аудио (macOS)")
    args = parser.parse_args()

    config.mock_audio = args.mock or not args.capture
    config.debug = args.debug
    config.setup()

    print("=" * 50)
    print("🎙  Meeting Copilot (MVP)")
    print(f"🎤  Whisper: {config.stt.model_size} ({config.stt.device})")
    print(f"🤖  LLM: {config.llm.model} @ {config.llm.base_url}")
    print(f"📋  Контекст: {config.context.max_seconds // 60} мин")
    if config.mock_audio:
        print("⚡  DEV MODE (mock audio)")
    print("=" * 50)

    # Инициализация
    context = ContextRing()
    llm = LlmClient()

    # Callback для транскрибации
    def on_text(text: str, source: str):
        context.add(text, source)
        if config.debug:
            print(f"[{'🎤' if source == 'mic' else '🔈'}] {text}")

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
        time.sleep(2)  # ждем старт UI

        def on_trigger():
            """Обработка триггера Cmd+Shift+H."""
            transcript = context.get_recent(15)
            if not transcript:
                overlay.show_result("Нет контекста для поиска")
                return

            overlay.show_status("Ищу в базе знаний...")

            # RAG поиск
            rag = None
            try:
                from rag_search import RagSearch
                rs = RagSearch()
                rag = rs.search(transcript[:200], limit=3)
            except Exception:
                pass

            # LLM запрос
            overlay.show_status("Анализирую...")
            messages = context.get_llm_messages(15)
            response = llm.ask(
                user_message="Проанализируй контекст разговора и дай короткую подсказку по NetSuite, если это уместно.",
                context=messages,
                rag_context=rag,
            )
            overlay.show_result(response)

            if config.debug:
                print(f"\n[Ответ] {response}\n")

        hotkey = HotkeyListener(on_trigger=on_trigger)
        hotkey.start()

    # Main loop — кормим аудио в транскрайбер
    print("\n▶  Running. Press Ctrl+C to stop.\n")

    try:
        last_print = 0
        while True:
            chunk = audio.read(timeout=1.0)
            if chunk:
                transcriber.feed(chunk.data, chunk.source)

            # Показываем статус каждые 30 секунд
            now = time.time()
            if config.debug and now - last_print > 30:
                dur = context.duration_seconds
                n = len(context._entries)
                p = transcriber.pending
                print(f"[Status] {n} фраз, {dur:.0f}s контекста, pending STT: {p}")
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

if __name__ == "__main__":
    sys.exit(main())
