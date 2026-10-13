#!/usr/bin/env python3
"""unigram BLEU score"""

import numpy as np


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


def uni_bleu(references, sentence):
    """calculate the unigram BLEU score of a sentence

    references is the list of reference translations, each one a list
    of the words in that translation
    sentence is the list of words the model came up with
    returns the unigram BLEU score
    """
    counts = {}
    for word in sentence:
        counts[word] = counts.get(word, 0) + 1
    clipped = 0
    for word, count in counts.items():
        best = 0
        for ref in references:
            best = max(best, ref.count(word))
        clipped += min(count, best)
    precision = clipped / len(sentence)
    ref_len = closest_ref_len(references, len(sentence))
    if len(sentence) > ref_len:
        penalty = 1
    else:
        penalty = np.exp(1 - ref_len / len(sentence))
    return float(precision * penalty)
