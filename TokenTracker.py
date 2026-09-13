"""
Модуль TokenTracker.py
Отвечает за подсчёт токенов, оценку стоимости и мониторинг лимитов контекста.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import tiktoken

from Ui import print_info, print_warning


# Лимиты контекста моделей DeepSeek (в токенах)
MODEL_CONTEXT_LIMITS = {
    "deepseek-reasoner": 64_000,
    "deepseek-chat": 64_000,
    "deepseek-v4-flash": 64_000,
    "deepseek-v4-pro": 128_000,
}

# Цены DeepSeek за 1M токенов (USD) — актуальные на сентябрь 2026
MODEL_PRICES = {
    "deepseek-reasoner": {"input": 0.55, "output": 2.19, "reasoning": 2.19},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input": 0.55, "output": 1.10},
}


@dataclass
class RequestStats:
    """Статистика одного запроса."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass
class TokenTracker:
    """
    Трекер токенов. Считает токены по каждому запросу и суммарно,
    оценивает стоимость и предупреждает о приближении к лимиту.
    """
    model: str
    requests: List[RequestStats] = field(default_factory=list)
    _encoder: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Используем cl100k_base — близок к токенизатору DeepSeek
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def estimate_tokens(self, text: str) -> int:
        """Оценивает количество токенов в тексте (для предварительной проверки)."""
        if self._encoder is None:
            # Грубая оценка: ~4 символа на токен для английского, ~2 для русского
            return len(text) // 3
        return len(self._encoder.encode(text))

    def estimate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Оценивает токены в массиве messages (с накладными расходами)."""
        # Каждое сообщение имеет ~4 токена накладных расходов (role, separators)
        total = sum(self.estimate_tokens(m.get("content", "")) + 4 for m in messages)
        return total + 3  # +3 на priming tokens

    def record_request(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
    ) -> RequestStats:
        """Записывает статистику выполненного запроса."""
        stats = RequestStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self.requests.append(stats)
        return stats

    def get_total_prompt(self) -> int:
        return sum(r.prompt_tokens for r in self.requests)

    def get_total_completion(self) -> int:
        return sum(r.completion_tokens for r in self.requests)

    def get_total_reasoning(self) -> int:
        return sum(r.reasoning_tokens for r in self.requests)

    def get_total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.requests)

    def get_cost(self) -> float:
        """Считает суммарную стоимость всех запросов в USD."""
        prices = MODEL_PRICES.get(self.model, {"input": 0.5, "output": 1.5})
        input_cost = self.get_total_prompt() * prices["input"] / 1_000_000
        output_cost = self.get_total_completion() * prices["output"] / 1_000_000
        # Для reasoner reasoning-токены тарифицируются как output
        if "reasoning" in prices:
            output_cost += self.get_total_reasoning() * prices["reasoning"] / 1_000_000
        return input_cost + output_cost

    def get_context_limit(self) -> int:
        return MODEL_CONTEXT_LIMITS.get(self.model, 64_000)

    def check_context_overflow(self, messages: List[Dict[str, str]]) -> bool:
        """Проверяет, не превысит ли запрос лимит контекста."""
        estimated = self.estimate_messages_tokens(messages)
        limit = self.get_context_limit()
        if estimated > limit * 0.9:  # Предупреждаем при 90%
            print_warning(
                f"⚠️ Приближение к лимиту контекста: "
                f"~{estimated} / {limit} токенов ({estimated * 100 // limit}%)"
            )
            return True
        return False

    def get_stats(self) -> Dict:
        """Возвращает полную статистику."""
        return {
            "model": self.model,
            "requests_count": len(self.requests),
            "total_prompt_tokens": self.get_total_prompt(),
            "total_completion_tokens": self.get_total_completion(),
            "total_reasoning_tokens": self.get_total_reasoning(),
            "total_tokens": self.get_total_tokens(),
            "total_cost_usd": self.get_cost(),
            "context_limit": self.get_context_limit(),
        }

    def print_stats(self) -> None:
        """Красиво выводит статистику в консоль."""
        stats = self.get_stats()
        print_info("=" * 50)
        print_info(f"📊 СТАТИСТИКА ТОКЕНОВ | Модель: {stats['model']}")
        print_info("=" * 50)
        print_info(f"🔢 Запросов выполнено:   {stats['requests_count']}")
        print_info(f"📥 Входных токенов:      {stats['total_prompt_tokens']:,}")
        print_info(f"📤 Выходных токенов:     {stats['total_completion_tokens']:,}")
        if stats['total_reasoning_tokens'] > 0:
            print_info(f"🧠 Рассуждений:          {stats['total_reasoning_tokens']:,}")
        print_info(f"🎯 Всего токенов:        {stats['total_tokens']:,}")
        print_info(f"💰 Стоимость:            ${stats['total_cost_usd']:.6f}")
        print_info(f"📏 Лимит контекста:      {stats['context_limit']:,}")
        print_info("=" * 50)