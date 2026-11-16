import random

import pyperclip

topics = []
with open("topics.txt") as f:
    for line in f:
        topics.append(line.strip())
random.shuffle(topics)
choice = random.choice(topics)
pyperclip.copy(choice)
print(pyperclip.paste())
