#!/usr/bin/env python3
"""handing a gensim model over to keras"""


def gensim_to_keras(model):
    """turn a trained gensim word2vec model into a keras layer

    model is the trained gensim word2vec model
    returns the trainable keras Embedding layer
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
