import time
from Agent import AgentConfig, DeepSeekAgent, print_info, print_critical, Color

def main() -> None:
    """Точка входа: демонстрация работы агента в интерактивном режиме."""

    # Конфигурация агента (можно легко менять)
    agent_config = AgentConfig(
        model="deepseek-reasoner",  # Или "deepseek-v4-pro", "deepseek-chat"
        temperature=0.7,
        reasoning_effort="high"
    )

    try:
        # Создаем экземпляр агента (отдельная сущность)
        agent = DeepSeekAgent(config=agent_config)
        print_info("Агент успешно инициализирован и готов к работе.")
        print(f"{Color.BOLD}Введите 'выход' для завершения, 'очистить' для сброса памяти.{Color.RESET}\n")

        # Интерактивный цикл (Чат)
        while True:
            quest = input(f"{Color.BOLD}Вы: {Color.RESET}").strip()

            if not quest:
                continue
            if quest.lower() in ["выход", "exit", "quit"]:
                print_info("Завершение работы агента.")
                break
            if quest.lower() in ["очистить", "clear"]:
                agent.clear_memory()
                continue

            start_time = time.perf_counter()

            # Агент обрабатывает запрос (вся логика инкапсулирована внутри)
            result = agent.process_request(user_message=quest)

            elapsed = time.perf_counter() - start_time

            if result["success"]:
                print(f"\n{Color.BOLD}=== 🧠 Рассуждения агента ==={Color.RESET}")
                print(result["reasoning"] or "[Скрыто или не требуется для данной модели]")

                print(f"\n{Color.BOLD}=== 💬 Ответ агента ==={Color.RESET}")
                print(result["content"] or "[Контент не возвращён]")
                print(f"{Color.YELLOW}⏱ Затраченное время: {elapsed:.2f} сек.{Color.RESET}\n")
            else:
                print_critical("Агент не смог сформировать ответ.")

    except EnvironmentError as e:
        print_critical(f"Ошибка конфигурации: {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print_info("\nРабота прервана пользователем.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()