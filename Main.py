"""
Главный модуль. Точка входа в приложение.
"""

import time

from Agent import AgentConfig, DeepSeekAgent
from Memory import MemoryManager
from TokenTracker import TokenTracker
from Ui import print_info, print_critical, Color


def main() -> None:
    agent_config = AgentConfig(
        model="deepseek-reasoner",
        temperature=0.7,
        reasoning_effort="high"
    )

    memory = MemoryManager(storage_path="chat_history.json")
    token_tracker = TokenTracker(model=agent_config.model)

    try:
        agent = DeepSeekAgent(
            config=agent_config,
            memory=memory,
            token_tracker=token_tracker,
        )
        print_info("🤖 Агент инициализирован. Память восстановлена.")
        print(f"{Color.BOLD}Команды: 'выход', 'очистить', 'история', 'статистика'{Color.RESET}\n")

        while True:
            quest = input(f"{Color.BOLD}Вы: {Color.RESET}").strip()

            if not quest:
                continue
            if quest.lower() in ["выход", "exit", "quit"]:
                print_info("Завершение работы агента.")
                token_tracker.print_stats()  # Финальная статистика
                break
            if quest.lower() in ["очистить", "clear"]:
                agent.clear_memory()
                continue
            if quest.lower() in ["история", "history"]:
                print(f"\n{Color.BOLD}=== 📜 История ({len(memory)} сообщ.) ==={Color.RESET}")
                for msg in memory.get_history():
                    prefix = "🧑" if msg["role"] == "user" else "🤖"
                    print(f"{prefix} {msg['content'][:120]}...")
                continue
            if quest.lower() in ["статистика", "stats"]:
                token_tracker.print_stats()
                continue

            start_time = time.perf_counter()
            result = agent.process_request(user_message=quest)
            elapsed = time.perf_counter() - start_time

            if result["success"]:
                print(f"\n{Color.BOLD}=== 🧠 Рассуждения ==={Color.RESET}")
                print(result["reasoning"] or "[скрыто]")
                print(f"\n{Color.BOLD}=== 💬 Ответ ==={Color.RESET}")
                print(result["content"] or "[нет ответа]")
                print(
                    f"{Color.YELLOW}⏱ Время: {elapsed:.2f} сек. | "
                    f"Токены: {result['prompt_tokens']:,} → {result['completion_tokens']:,}"
                    f"{Color.RESET}\n"
                )
            else:
                print_critical("Агент не смог сформировать ответ.")

    except EnvironmentError as e:
        print_critical(f"Ошибка конфигурации: {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print_info("\nРабота прервана.")
        token_tracker.print_stats()
        raise SystemExit(0)


if __name__ == "__main__":
    main()