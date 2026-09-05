"""
Клиент DeepSeek Reasoning API

Production-ready обёртка для эндпоинта chat completions DeepSeek
с поддержкой режима рассуждения. Обрабатывает аутентификацию,
восстановление после ошибок и безопасный парсинг ответов.
"""

import os
import logging
from typing import Optional

# ✅ load_dotenv() ВЫЗЫВАЕТСЯ ПЕРВОЙ, до любых обращений к os.environ
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError

# Настройка структурированного логирования вместо print
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
            "Переменная окружения DEEPSEEK_API_KEY не установлена. "
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
    model: str = "deepseek-reasoner",
    reasoning_effort: str = "high",
) -> dict[str, Optional[str]]:
    """
    Отправка запроса на генерацию ответа через DeepSeek с включённым режимом рассуждения.

    Args:
        user_message: Входной промпт пользователя.
        system_prompt: Системная инструкция для модели.
        model: Идентификатор модели. Используйте 'deepseek-reasoner' для R1.
        reasoning_effort: Глубина рассуждения ('low', 'medium', 'high').

    Returns:
        Словарь с ключами 'content' (финальный ответ) и 'reasoning' (цепочка рассуждений).

    Raises:
        APIError: При неустранимых ошибках API после логирования.
    """
    client = create_deepseek_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            reasoning_effort=reasoning_effort,
            extra_body={"thinking": {"type": "enabled"}},
        )

        choice = response.choices[0]
        message = choice.message

        result = {
            "content": getattr(message, "content", None),
            "reasoning": getattr(message, "reasoning_content", None),
            "finish_reason": choice.finish_reason,
        }

        logger.info(
            "Ответ получен | finish_reason=%s | есть_рассуждение=%s",
            result["finish_reason"],
            result["reasoning"] is not None,
        )
        return result

    except RateLimitError as e:
        logger.error("Превышен лимит запросов. Внедрите backoff/retry. Детали: %s", e)
        raise
    except APITimeoutError as e:
        logger.error("Таймаут запроса. Увеличьте timeout или снизьте reasoning_effort. Детали: %s", e)
        raise
    except APIConnectionError as e:
        logger.error("Ошибка сетевого подключения. Проверьте соединение и base_url. Детали: %s", e)
        raise
    except APIError as e:
        logger.error("Ошибка DeepSeek API | статус=%s | сообщение=%s", e.status_code, e.message)
        raise


def main() -> None:
    """Точка входа с демонстрацией использования и корректной обработкой ошибок."""
    try:
        result = get_reasoned_completion(user_message="Привет")

        if result["reasoning"]:
            print("=== Процесс мышления ===")
            print(result["reasoning"])
            print("========================\n")

        print("=== Ответ ===")
        print(result["content"] or "[Контент не возвращён]")

    except EnvironmentError as e:
        logger.critical("Ошибка конфигурации: %s", e)
        raise SystemExit(1)
    except APIError:
        raise SystemExit(1)


if __name__ == "__main__":
    main()