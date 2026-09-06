import sys


class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_info(message: str) -> None:
    print(f"{Color.GREEN}[INFO]{Color.RESET} {message}")


def print_warning(message: str) -> None:
    print(f"{Color.YELLOW}[WARN]{Color.RESET} {message}", file=sys.stderr)


def print_error(message: str) -> None:
    print(f"{Color.RED}[ERROR]{Color.RESET} {message}", file=sys.stderr)


def print_critical(message: str) -> None:
    print(f"{Color.RED}{Color.BOLD}[CRITICAL]{Color.RESET} {message}", file=sys.stderr)