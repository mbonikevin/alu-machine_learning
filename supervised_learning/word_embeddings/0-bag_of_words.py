#!/usr/bin/env python3
"""bag of words embedding"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """create a bag of words embedding matrix

    sentences is the list of sentences to analyze
    vocab is the list of words to use, all of the words in the
    sentences are used when it is None
    returns the (s, f) embedding matrix and the list of features
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    return embeddings, list(get_features(vectorizer))


def get_features(vectorizer):
    """read the feature names off a fitted vectorizer

    the method was renamed in the newer versions of scikit learn
    """
    if hasattr(vectorizer, 'get_feature_names_out'):
        return vectorizer.get_feature_names_out()
    return vectorizer.get_feature_names()
