#!/usr/bin/env python3
"""answer questions out of one reference document"""

question_answer = __import__('0-qa').question_answer

EXITS = ['exit', 'quit', 'goodbye', 'bye']


def answer_loop(reference):
    """keep asking for questions until the user says goodbye

    reference is the string with the document to look in
    """
    while True:
        question = input('Q: ')
        if question.strip().lower() in EXITS:
            print('A: Goodbye')
            break
        answer = question_answer(question, reference)
        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
