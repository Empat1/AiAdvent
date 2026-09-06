"""
Клиент DeepSeek Reasoning API

Консольная обёртка для эндпоинта chat completions DeepSeek
с поддержкой режима рассуждения. Обрабатывает аутентификацию,
восстановление после ошибок и безопасный парсинг ответов.
"""

import os
import time

from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from Ui import *

# ✅ Загрузка переменных окружения ДО любых обращений к os.environ
load_dotenv()

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError

TASK = """У вас есть 25 лошадей и трасса на 5 лошадей. 
Секундомера нет — виден только порядок финиша в каждом забеге.
Какое минимальное количество забегов нужно, чтобы гарантированно 
найти 3 самых быстрых лошади?

Ответь: число забегов + доказательство, 
что меньшим числом забегов обойтись нельзя."""

@dataclass
class ModelConfig:
    name: str


MODELS = [
    ModelConfig("deepseek-v4-flash"),
    ModelConfig("deepseek-v4-pro"),
    ModelConfig("deepseek-chat"),
    ModelConfig("deepseek-reasoner"),
]


def create_deepseek_client() -> OpenAI:
    """
    Фабричная функция для создания аутентифицированного клиента DeepSeek.

    Returns:
        Сконфигурированный OpenAI-совместимый клиент, указывающий на DeepSeek API.

    Raises:
        EnvironmentError: Если DEEPSEEK_API_KEY не задан или пуст.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Переменная окружения DEEPSEEK_API_KEY не установлена.\n"
            "Создайте файл .env с ключом или экспортируйте переменную."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=120.0,
    )


def get_reasoned_completion(
    user_message: str,
    system_prompt: str = "Вы — полезный ассистент.",
    model: str = "deepseek-v4-flash",
    reasoning_effort: str = "high",
    need_think: bool = False,
) -> dict[str, Optional[str]]:
    client = create_deepseek_client()

    print_info(f"Отправка запроса к модели '{model}'")
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    if need_think:
        params["extra_body"] = {"thinking": {"type": "enabled"}}
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**params)

    choice = response.choices[0]
    message = choice.message

    result = {
        "content": getattr(message, "content", None),
        "reasoning": getattr(message, "reasoning_content", None),
        "finish_reason": choice.finish_reason,
    }

    has_reasoning = result["reasoning"] is not None
    print_info(f"Ответ получен | finish_reason={result['finish_reason']} ")
    return result

def handle_api_errors(func):
    """Декоратор для обработки ошибок DeepSeek API"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            print_error(f"Превышен лимит запросов. Внедрите backoff/retry. Детали: {e}")
            raise
        except APITimeoutError as e:
            print_error(f"Таймаут запроса. Увеличьте timeout или снизьте reasoning_effort. Детали: {e}")
            raise
        except APIConnectionError as e:
            print_error(f"Ошибка сетевого подключения. Проверьте соединение и base_url. Детали: {e}")
            raise
        except APIError as e:
            print_error(f"Ошибка DeepSeek API | статус={e.status_code} | сообщение={e.message}")
            raise
    return wrapper


def main() -> None:
    """Точка входа с консольным выводом и обработкой ошибок."""
    try:
        quest = input("Запрос пользователя ")

        for model in MODELS:
            start_time = time.perf_counter()
            result1 = get_reasoned_completion(user_message=quest, model=model.name)
            print(f"{Color.BOLD}=== Ответ ==={Color.RESET}")
            print(result1["content"] or "[Контент не возвращён]")

            elapsed = time.perf_counter() - start_time

            print(f"{Color.BOLD}=== Ответ ==={Color.RESET}")
            print(result1["content"] or "[Контент не возвращён]")
            print(f"Затраченное время = {elapsed}")

    except EnvironmentError as e:
        print_critical(f"Ошибка конфигурации: {e}")
        raise SystemExit(1)
    except APIError:
        # Уже выведено в консоль внутри get_reasoned_completion
        raise SystemExit(1)


if __name__ == "__main__":
    main()