"""Pick a random topic from topics.txt and copy it to the clipboard."""

from __future__ import annotations

import random
import sys

import pyperclip


def main() -> None:
    with open("topics.txt", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]
    if not topics:
        sys.exit("topics.txt is empty.")
    choice = random.choice(topics)
    pyperclip.copy(choice)
    print(choice)


if __name__ == "__main__":
    main()
