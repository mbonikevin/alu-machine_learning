#!/usr/bin/env python3
"""tf-idf embedding"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """create a tf-idf embedding matrix

    sentences is the list of sentences to analyze
    vocab is the list of words to use, all of the words in the
    sentences are used when it is None
    returns the (s, f) embedding matrix and the list of features
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    return embeddings, list(get_features(vectorizer))


def get_features(vectorizer):
    """read the feature names off a fitted vectorizer

    the method was renamed in the newer versions of scikit learn
    """
    if hasattr(vectorizer, 'get_feature_names_out'):
        return vectorizer.get_feature_names_out()
    return vectorizer.get_feature_names()
