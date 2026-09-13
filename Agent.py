"""
Модуль Agent.py
Инкапсулирует логику работы с DeepSeek API, память и подсчёт токенов.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from openai import (
    OpenAI, APIError, APITimeoutError, APIConnectionError,
    RateLimitError, BadRequestError,
)

from Ui import print_info, print_warning, print_error, print_critical, Color
from Memory import MemoryManager
from TokenTracker import TokenTracker


@dataclass
class AgentConfig:
    model: str
    temperature: float
    reasoning_effort: str = "high"
    system_prompt: str = "Вы — полезный ассистент, способный к глубоким рассуждениям."


class DeepSeekAgent:
    """Автономный агент с персистентной памятью и трекером токенов."""

    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryManager,
        token_tracker: TokenTracker,
    ):
        self.config = config
        self.memory = memory
        self.token_tracker = token_tracker
        self.client = self._create_client()
        self.max_retries = 3

    def _create_client(self) -> OpenAI:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY не установлена.")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=120.0)

    def _build_messages(self, user_message: str) -> list:
        """Формирует messages: system + история + новый запрос."""
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.memory.get_history())
        messages.append({"role": "user", "content": user_message})
        return messages

    def _trim_history_if_needed(self, messages: list) -> list:
        """
        Если история превышает лимит контекста — обрезает старые сообщения,
        сохраняя system prompt и последний запрос.
        """
        limit = self.token_tracker.get_context_limit()
        estimated = self.token_tracker.estimate_messages_tokens(messages)

        if estimated <= limit:
            return messages

        print_warning(
            f"🔥 Превышен лимит контекста ({estimated} > {limit}). "
            f"Обрезаю старые сообщения..."
        )

        system_msg = messages[0]
        user_msg = messages[-1]
        history = messages[1:-1]

        # Удаляем самые старые сообщения парами (user + assistant)
        while len(history) > 2 and self.token_tracker.estimate_messages_tokens(
            [system_msg] + history + [user_msg]
        ) > limit * 0.85:
            history = history[2:]  # удаляем пару

        return [system_msg] + history + [user_msg]

    def compress_history_if_needed(self, trigger_threshold: int = 10, keep_last_n: int = 5) -> None:
        """
        Проверяет длину истории и сжимает её, если превышен порог.
        """
        to_compress, keep_raw = self.memory.get_compressible_history(keep_last_n=keep_last_n)

        if not to_compress:
            return  # Сжимать нечего

        print_info(f"🔄 Превышен порог истории ({trigger_threshold}+). Запускаю сжатие контекста...")

        # Форматируем текст для сжатия
        text_to_summarize = "\n".join([f"{m['role']}: {m['content']}" for m in to_compress])

        summary_prompt = (
            "Сделай предельно краткое резюме этого фрагмента диалога. "
            "Сохрани ВСЕ важные факты, имена, числа, договоренности и контекст. "
            "Убери воду, приветствия и повторяющиеся мысли. "
            "Максимум 3-4 предложения.\n\n"
            f"Диалог:\n{text_to_summarize}"
        )

        try:
            # 🔑 КЛЮЧЕВОЙ МОМЕНТ: используем ДЕШЁВУЮ модель для сжатия!
            # Не тратьте дорогой reasoner на саммари.
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # или "deepseek-v4-flash"
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2,
                max_tokens=512,
                stream=False
            )

            new_summary = response.choices[0].message.content
            self.memory.apply_summary(new_summary, keep_raw)

        except Exception as e:
            print_warning(f"Не удалось сжать историю: {e}. Продолжаю с полной историей.")

    def process_request(self, user_message: str) -> Dict[str, Optional[Any]]:
        messages = self._build_messages(user_message)

        # 🔍 Предварительная оценка токенов
        estimated_tokens = self.token_tracker.estimate_messages_tokens(messages)
        print_info(f"🔢 Оценка токенов запроса: ~{estimated_tokens:,}")

        # ⚠️ Проверка на переполнение
        self.token_tracker.check_context_overflow(messages)

        # ✂️ Обрезка истории при необходимости
        messages = self._trim_history_if_needed(messages)

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

                # 🔑 Извлекаем РЕАЛЬНУЮ статистику токенов из ответа API
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                # Для reasoner reasoning_tokens приходят в отдельном поле
                reasoning_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details"):
                    details = usage.completion_tokens_details
                    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

                # 🔑 Записываем в трекер
                self.token_tracker.record_request(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                )

                # Сохраняем в персистентную память
                self.memory.add_message("user", user_message)
                self.memory.add_message("assistant", content)

                # 📊 Показываем статистику текущего запроса
                print_info(
                    f"✅ Токены запроса: "
                    f"вход={prompt_tokens:,}, выход={completion_tokens:,}"
                    + (f", рассуждения={reasoning_tokens:,}" if reasoning_tokens else "")
                )

                return {
                    "content": content,
                    "reasoning": reasoning,
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "success": True,
                }

            except BadRequestError as e:
                # 🔥 КРИТИЧНО: обработка переполнения контекста
                if "context_length" in str(e).lower() or "too long" in str(e).lower():
                    print_critical(
                        f"💥 КОНТЕКСТ ПЕРЕПОЛНЁН! {e}\n"
                        f"   Лимит модели: {self.token_tracker.get_context_limit():,} токенов"
                    )
                    # Пытаемся обрезкой спасти ситуацию
                    messages = self._trim_history_if_needed(messages)
                    if len(messages) <= 2:  # только system + user
                        print_critical("История уже минимальна. Очистите память командой 'очистить'.")
                        break
                    print_info("🔄 Повторяю запрос с обрезанной историей...")
                    params["messages"] = messages
                    continue
                print_error(f"Ошибка запроса: {e}")
                break

            except (APIConnectionError, APITimeoutError) as e:
                wait_time = 2 ** attempt
                print_warning(f"Сетевая ошибка: {e}. Повтор через {wait_time} сек...")
                time.sleep(wait_time)

            except RateLimitError:
                print_warning("Превышен лимит запросов. Ожидание 5 секунд...")
                time.sleep(5)

            except APIError as e:
                print_error(f"Ошибка API: {e}")
                break

        print_critical("Не удалось получить ответ после нескольких попыток.")
        return {"content": None, "reasoning": None, "finish_reason": "error", "success": False}

    def clear_memory(self) -> None:
        self.memory.clear()
