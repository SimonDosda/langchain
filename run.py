#!/usr/bin/env python3
"""
Script runner for LangChain demos.
Usage: python run.py <demo_name>
Example: python run.py translation
"""

import sys
from typing import Dict, Callable

def run_translation() -> None:
    from scripts.translation_demo import main
    main()

def run_chatbot() -> None:
    from scripts.chatbot_demo import main
    main()

AVAILABLE_DEMOS: Dict[str, Callable[[], None]] = {
    "translation": run_translation,
    "chatbot": run_chatbot,
}

def print_usage() -> None:
    print("Available demos:")
    for demo in AVAILABLE_DEMOS:
        print(f"  - {demo}")
    print("\nUsage: python run.py <demo_name>")

def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ["-h", "--help"]:
        print_usage()
        return

    demo_name = sys.argv[1]
    if demo_name not in AVAILABLE_DEMOS:
        print(f"Error: Unknown demo '{demo_name}'")
        print_usage()
        return

    AVAILABLE_DEMOS[demo_name]()

if __name__ == "__main__":
    main() 