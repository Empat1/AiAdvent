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

    def get_compressible_history(self, keep_last_n: int = 5) -> tuple[list, list]:
        """
        Разделяет историю на две части:
        1. Сообщения для сжатия (всё, кроме последних N)
        2. Последние N сообщений (оставляем как есть)
        """
        history = self.get_history()
        if len(history) <= keep_last_n:
            return [], history

        # Ищем, есть ли уже старое резюме, чтобы не потерять его
        # Мы будем хранить резюме как сообщение с ролью "summary"
        summary_msg = None
        raw_history = []

        for msg in history:
            if msg.get("role") == "system":
                summary_msg = msg
            else:
                raw_history.append(msg)

        if len(raw_history) <= keep_last_n:
            return [], history

        to_compress = raw_history[:-keep_last_n]
        keep_raw = raw_history[-keep_last_n:]

        # Если было старое резюме, добавляем его в начало сжимаемого блока
        if summary_msg:
            to_compress = [summary_msg] + to_compress

        return to_compress, keep_raw

    def apply_summary(self, summary_text: str, recent_messages: list) -> None:
        """
        Заменяет старую историю на резюме + последние сообщения.
        """
        new_history = [
            {"role": "system", "content": f"КРАТКОЕ РЕЗЮМЕ ПРЕДЫДУЩЕГО ДИАЛОГА:\n{summary_text}"}
        ]
        new_history.extend(recent_messages)

        self.history = new_history
        self._save()
        print_info("🗜 История сжата. Старые сообщения заменены на резюме.")