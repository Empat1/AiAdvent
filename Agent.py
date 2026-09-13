"""
Модуль Agent.py
Инкапсулирует логику работы с DeepSeek API + интеграцию с MemoryManager.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError

from Ui import print_info, print_warning, print_error, print_critical, Color
from Memory import MemoryManager


@dataclass
class AgentConfig:
    model: str
    temperature: float
    reasoning_effort: str = "high"
    system_prompt: str = "Вы — полезный ассистент, способный к глубоким рассуждениям."


class DeepSeekAgent:
    """Автономный агент с персистентной памятью."""

    def __init__(self, config: AgentConfig, memory: MemoryManager):
        self.config = config
        self.memory = memory
        self.client = self._create_client()
        self.max_retries = 3

    def _create_client(self) -> OpenAI:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY не установлена.")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=120.0)

    def _build_messages(self, user_message: str) -> list:
        """Формирует массив messages для API: system + история + новый запрос."""
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.memory.get_history())
        messages.append({"role": "user", "content": user_message})
        return messages

    def process_request(self, user_message: str) -> Dict[str, Optional[Any]]:
        messages = self._build_messages(user_message)

        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }

        if "reasoner" in self.config.model or "v4" in self.config.model:
            params["extra_body"] = {"thinking": {"type": "enabled"}}
            params["reasoning_effort"] = self.config.reasoning_effort

        for attempt in range(self.max_retries):
            try:
                print_info(f"Запрос к '{self.config.model}' (Попытка {attempt + 1}/{self.max_retries})...")
                response = self.client.chat.completions.create(**params)

                choice = response.choices[0]
                message = choice.message
                content = getattr(message, "content", None)
                reasoning = getattr(message, "reasoning_content", None)

                # Сохраняем и запрос, и ответ в персистентную память
                self.memory.add_message("user", user_message)
                self.memory.add_message("assistant", content)

                print_info(f"Ответ получен | finish_reason={choice.finish_reason}")
                return {
                    "content": content,
                    "reasoning": reasoning,
                    "finish_reason": choice.finish_reason,
                    "success": True,
                }

            except (APIConnectionError, APITimeoutError) as e:
                # Сетевые ошибки — не критичны, если ретраи помогают
                wait_time = 2 ** attempt
                print_warning(f"Сетевая ошибка: {e}. Повтор через {wait_time} сек...")
                time.sleep(wait_time)

            except RateLimitError:
                print_warning("Превышен лимит запросов. Ожидание 5 секунд...")
                time.sleep(5)

            except APIError as e:
                # Ошибки API — серьёзные, повторяем бессмысленно
                print_error(f"Ошибка API: {e}")
                break

        print_critical("Не удалось получить ответ после нескольких попыток.")
        return {"content": None, "reasoning": None, "finish_reason": "max_retries_exceeded", "success": False}

    def clear_memory(self) -> None:
        self.memory.clear()