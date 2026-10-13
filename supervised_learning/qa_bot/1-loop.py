#!/usr/bin/env python3
"""a loop that takes questions and answers nothing yet"""

EXITS = ['exit', 'quit', 'goodbye', 'bye']

while True:
    question = input('Q: ')
    if question.strip().lower() in EXITS:
        print('A: Goodbye')
        break
    print('A:')
