"""Глобальный хоткей + overlay UI (PyQt6)."""

import sys
import threading
from typing import Optional, Callable
from config import config


class HotkeyListener:
    """Слушатель глобального хоткея (pynput для macOS)."""

    def __init__(self, on_trigger: Optional[Callable[[], None]] = None):
        self.on_trigger = on_trigger
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._listener = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print(f"Hotkey active: {config.overlay.hotkey}")

    def stop(self):
        self._running = False
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass

    def _listen(self):
        try:
            from pynput import keyboard as pynput_kb

            # Парсим hotkey: "cmd+shift+h" → [Key.cmd, Key.shift, 'h']
            parts = config.overlay.hotkey.split("+")
            key_map = {
                "cmd": pynput_kb.Key.cmd,
                "command": pynput_kb.Key.cmd,
                "ctrl": pynput_kb.Key.ctrl,
                "control": pynput_kb.Key.ctrl,
                "shift": pynput_kb.Key.shift,
                "alt": pynput_kb.Key.alt,
                "option": pynput_kb.Key.alt,
            }

            modifiers = set()
            target_key = None

            for p in parts:
                p = p.strip().lower()
                if p in key_map:
                    modifiers.add(key_map[p])
                elif len(p) == 1:
                    target_key = pynput_kb.KeyCode.from_char(p)

            if target_key is None:
                print("[Hotkey] Не удалось распарсить хоткей")
                self._listen_fallback()
                return

            current = set()

            def on_press(key):
                if key in modifiers:
                    current.add(key)
                elif key == target_key and current == modifiers:
                    if self.on_trigger:
                        self.on_trigger()

            def on_release(key):
                current.discard(key)

            self._listener = pynput_kb.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
            self._listener.wait()

        except ImportError:
            print("[Hotkey] pynput не установлен — fallback на Enter")
            self._listen_fallback()
        except Exception as e:
            print(f"[Hotkey] Ошибка: {e} — fallback на Enter")
            self._listen_fallback()

    def _listen_fallback(self):
        """Fallback: Enter в консоли (для dev/отладки)."""
        print("[Hotkey fallback] Press Enter to trigger")
        while self._running:
            try:
                input()
                if self.on_trigger:
                    self.on_trigger()
            except (EOFError, KeyboardInterrupt):
                break


class OverlayUI:
    """Плавающее окно с подсказками (PyQt6)."""

    def __init__(self):
        self._app = None
        self._window = None
        self._label = None
        self._started = threading.Event()

    def run(self):
        """Запуск UI в фоновом потоке."""
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        self._started.wait(timeout=5)

    def _run_loop(self):
        from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QFont

        app = QApplication(sys.argv)
        self._app = app

        overlay_cfg = config.overlay
        window = QMainWindow()
        self._window = window

        # Настройки окна
        window.setWindowTitle("Meeting Copilot")
        window.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        window.setFixedSize(overlay_cfg.width, overlay_cfg.height)

        # Виджет
        central = QWidget()
        central.setStyleSheet(f"""
            background: rgba(13, 17, 23, {overlay_cfg.opacity});
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 8px;
        """)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)

        self._label = QLabel("Готов к работе\nCmd+Shift+H — поиск")
        self._label.setWordWrap(True)
        self._label.setFont(QFont("SF Mono", overlay_cfg.font_size))
        self._label.setStyleSheet("color: #e6edf3; background: transparent;")
        layout.addWidget(self._label)

        window.setCentralWidget(central)

        # Позиция
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.right() - overlay_cfg.width - 20
            y = geom.bottom() - overlay_cfg.height - 60
            if "top" in overlay_cfg.position:
                y = 60
            if "center" in overlay_cfg.position:
                x = (geom.width() - overlay_cfg.width) // 2
            window.move(x, y)

        self._started.set()
        window.show()
        app.exec()

    def show_result(self, text: str):
        """Обновить текст в оверлее (thread-safe через QTimer)."""
        if self._label is None or self._app is None:
            return

        # Qt требует обновление UI из главного потока
        # Используем QTimer.singleShot для безопасного обновления
        def _update():
            if self._label:
                self._label.setText(text)

        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, _update)
        except Exception:
            # Если QTimer недоступен — напрямую (может быть небезопасно)
            _update()

    def show_status(self, text: str):
        """Показать статус (загрузка, ошибка)."""
        self.show_result(f"[{text}]")

    def stop(self):
        if self._app:
            self._app.quit()
