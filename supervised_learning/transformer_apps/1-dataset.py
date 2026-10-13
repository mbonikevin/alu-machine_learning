#!/usr/bin/env python3
"""dataset for portuguese to english translation"""

import tensorflow_datasets as tfds


class Dataset:
    """loads and preps the ted talk translation dataset"""

    def __init__(self):
        """load the training and the validation splits and tokenize"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """build a subword tokenizer for each language

        data is the dataset of (portuguese, english) sentence pairs
        returns the portuguese tokenizer and the english tokenizer
        """
        encoder = tfds.features.text.SubwordTextEncoder
        tokenizer_pt = encoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        tokenizer_en = encoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """turn a sentence pair into tokens

        the start of sentence token is the vocabulary size and the end
        of sentence token is the vocabulary size plus one
        returns the portuguese tokens and the english tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(pt.numpy())
        pt_tokens += [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(en.numpy())
        en_tokens += [self.tokenizer_en.vocab_size + 1]
        return pt_tokens, en_tokens
