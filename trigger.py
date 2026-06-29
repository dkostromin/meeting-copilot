"""Глобальный хоткей (Cmd+Shift+H) + overlay UI (PyQt6)."""

import sys
import threading
from typing import Optional, Callable
from config import config

class HotkeyListener:
    """Слушатель глобального хоткея.
    
    На macOS использует `keyboard` или fallback на input monitoring.
    """

    def __init__(self, on_trigger: Optional[Callable[[], None]] = None):
        self.on_trigger = on_trigger
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print(f"Hotkey active: {config.overlay.hotkey}")

    def stop(self):
        self._running = False

    def _listen(self):
        try:
            import keyboard
            hotkey = config.overlay.hotkey.replace("cmd", "command")
            keyboard.add_hotkey(hotkey, self._on_press)
            keyboard.wait()
        except (ImportError, NotImplementedError, OSError):
            # Fallback: polling stdin
            self._listen_fallback()

    def _listen_fallback(self):
        """Fallback: Enter в консоли (для dev/отладки)."""
        print("[Hotkey fallback] Press Enter to trigger")
        while self._running:
            try:
                input()
                self._on_press()
            except (EOFError, KeyboardInterrupt):
                break

    def _on_press(self):
        if self.on_trigger:
            self.on_trigger()


class OverlayUI:
    """Плавающее окно с подсказками (PyQt6)."""

    def __init__(self):
        self._app: Optional["QApplication"] = None  # noqa
        self._window: Optional["QWindow"] = None     # noqa
        self._started = threading.Event()

    def run(self):
        """Запуск UI в фоновом потоке."""
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        self._started.wait(timeout=5)

    def _run_loop(self):
        from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
        from PyQt6.QtCore import Qt, QRect, QPoint
        from PyQt6.QtGui import QFont, QScreen

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
        sys.exit(app.exec())

    def show_result(self, text: str):
        """Обновить текст в оверлее."""
        if self._label:
            self._label.setText(text)

    def show_status(self, text: str):
        """Показать статус (загрузка, ошибка)."""
        if self._label:
            self._label.setText(f"[{text}]")

    def stop(self):
        if self._app:
            self._app.quit()
