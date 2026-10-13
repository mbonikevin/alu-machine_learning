#!/usr/bin/env python3
"""question answering with a BERT model"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """find the snippet of the reference that answers the question

    question is the string with the question to answer
    reference is the string with the document to look in
    returns the answer, or None when nothing in the document answers it
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)
    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] +
              reference_tokens + ['[SEP]'])
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = ([0] * (len(question_tokens) + 2) +
                      [1] * (len(reference_tokens) + 1))
    input_word_ids, input_mask, input_type_ids = map(
        lambda ids: tf.expand_dims(
            tf.convert_to_tensor(ids, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    start = tf.argmax(outputs[0][0][1:]) + 1
    end = tf.argmax(outputs[1][0][1:]) + 1
    answer = tokenizer.convert_tokens_to_string(tokens[start:end + 1])
    if answer is None or answer.strip() == '':
        return None
    return answer
