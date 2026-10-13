#!/usr/bin/env python3
"""answer questions out of a whole corpus of documents"""

semantic_search = __import__('3-semantic_search').semantic_search
answer_question = __import__('0-qa').question_answer

EXITS = ['exit', 'quit', 'goodbye', 'bye']


def question_answer(corpus_path):
    """keep asking for questions and search the whole corpus

    corpus_path is the path to the folder of reference documents
    """
    while True:
        question = input('Q: ')
        if question.strip().lower() in EXITS:
            print('A: Goodbye')
            break
        reference = semantic_search(corpus_path, question)
        answer = answer_question(question, reference)
        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
