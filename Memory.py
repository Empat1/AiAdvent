"""
Модуль Memory.py
Персистентное хранение истории диалога агента в JSON.
"""

import json
from typing import List, Dict
from pathlib import Path

from Ui import print_info, print_critical


class MemoryManager:
    """Менеджер памяти агента. Сохраняет и загружает историю из JSON-файла."""

    def __init__(self, storage_path: str = "chat_history.json"):
        self.storage_path = Path(storage_path)
        self.history: List[Dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        """Загружает историю из файла при инициализации."""
        if not self.storage_path.exists():
            print_info(f"Файл истории не найден. Создаём новый: {self.storage_path}")
            self.history = []
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.history = data.get("messages", [])
            print_info(f"✅ Загружено сообщений из истории: {len(self.history)}")
        except (json.JSONDecodeError, OSError) as e:
            print_critical(f"Ошибка чтения истории: {e}. Начинаем с чистой памяти.")
            self.history = []

    def _save(self) -> None:
        """Сохраняет текущую историю в файл (UTF-8, читаемый формат)."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"messages": self.history}, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print_critical(f"Ошибка сохранения истории: {e}")

    def add_message(self, role: str, content: str) -> None:
        """Добавляет сообщение в историю и сразу сохраняет на диск."""
        self.history.append({"role": role, "content": content})
        self._save()

    def get_history(self) -> List[Dict[str, str]]:
        """Возвращает копию истории."""
        return list(self.history)

    def clear(self) -> None:
        """Очищает историю и удаляет файл."""
        self.history = []
        if self.storage_path.exists():
            self.storage_path.unlink()
        print_info("🗑 История полностью очищена.")

    def __len__(self) -> int:
        return len(self.history)