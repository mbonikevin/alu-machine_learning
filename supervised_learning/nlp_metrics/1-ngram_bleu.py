#!/usr/bin/env python3
"""n-gram BLEU score"""

import numpy as np


def ngrams(words, n):
    """cut a list of words into the list of its n-grams"""
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def closest_ref_len(references, length):
    """find the reference length that sits closest to the sentence

    a tie goes to the shorter reference
    """
    best = len(references[0])
    for ref in references:
        gap = abs(len(ref) - length)
        if gap < abs(best - length) or (gap == abs(best - length) and
                                        len(ref) < best):
            best = len(ref)
    return best


def ngram_bleu(references, sentence, n):
    """calculate the n-gram BLEU score of a sentence

    references is the list of reference translations, each one a list
    of the words in that translation
    sentence is the list of words the model came up with
    n is the size of the n-gram to use
    returns the n-gram BLEU score
    """
    candidate = ngrams(sentence, n)
    counts = {}
    for gram in candidate:
        counts[gram] = counts.get(gram, 0) + 1
    clipped = 0
    for gram, count in counts.items():
        best = 0
        for ref in references:
            best = max(best, ngrams(ref, n).count(gram))
        clipped += min(count, best)
    precision = clipped / len(candidate)
    ref_len = closest_ref_len(references, len(sentence))
    if len(sentence) > ref_len:
        penalty = 1
    else:
        penalty = np.exp(1 - ref_len / len(sentence))
    return float(precision * penalty)
