"""
Модуль Agent.py
Инкапсулирует всю логику работы с DeepSeek API, память и обработку ошибок.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

try:
    from Ui import print_info, print_critical, Color
except ImportError:
    class Color:
        BOLD = '\033[1m'
        RESET = '\033[0m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'

    def print_info(msg: str):
        print(f"{Color.GREEN}[INFO]{Color.RESET} {msg}")

    def print_critical(msg: str):
        print(f"{Color.RED}[CRITICAL]{Color.RESET} {msg}")

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError


@dataclass
class AgentConfig:
    """Конфигурация для агента."""
    model: str
    temperature: float
    reasoning_effort: str = "high"
    system_prompt: str = "Вы — полезный ассистент, способный к глубоким рассуждениям. Отвечайте структурированно."


class DeepSeekAgent:
    """Автономный агент для взаимодействия с DeepSeek API."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = self._create_client()
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        self.max_retries = 3

    def _create_client(self) -> OpenAI:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("Переменная окружения DEEPSEEK_API_KEY не установлена.")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=120.0)

    def process_request(self, user_message: str) -> Dict[str, Optional[Any]]:
        self.history.append({"role": "user", "content": user_message})

        params = {
            "model": self.config.model,
            "messages": self.history,
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

                self.history.append({"role": "assistant", "content": content})
                print_info(f"Ответ получен | finish_reason={choice.finish_reason}")

                return {"content": content, "reasoning": reasoning, "finish_reason": choice.finish_reason, "success": True}

            except (APIConnectionError, APITimeoutError) as e:
                wait_time = 2 ** attempt
                print_info(f"Сетевая ошибка: {e}. Повтор через {wait_time} сек...")
                time.sleep(wait_time)
            except RateLimitError:
                print_info("Превышен лимит запросов. Ожидание 5 секунд...")
                time.sleep(5)
            except APIError as e:
                print_critical(f"Критическая ошибка API: {e}")
                break

        print_critical("Не удалось получить ответ после нескольких попыток.")
        return {"content": None, "reasoning": None, "finish_reason": "max_retries_exceeded", "success": False}

    def clear_memory(self):
        self.history = [{"role": "system", "content": self.config.system_prompt}]
        print_info("Память агента очищена.")