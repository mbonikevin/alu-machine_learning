#!/usr/bin/env python3
"""training of a word2vec model"""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """create and train a gensim word2vec model

    sentences is the list of sentences to train on
    size is the size of the embedding layer
    min_count is how often a word has to show up to be trained
    window is the furthest the current and the predicted word can be
    negative is the size of the negative sampling
    cbow picks the training type, True is CBOW and False is skip-gram
    iterations is how many times to train over the sentences
    seed feeds the random number generator
    workers is the number of threads that train the model
    returns the trained model
    """
    model = Word2Vec(size=size, window=window, min_count=min_count,
                     negative=negative, sg=0 if cbow else 1, seed=seed,
                     workers=workers, iter=iterations)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
