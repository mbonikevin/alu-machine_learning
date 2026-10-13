#!/usr/bin/env python3
"""semantic search over a corpus of documents"""

import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """find the document that comes closest to the sentence

    corpus_path is the path to the folder of reference documents
    sentence is the string to match the documents against
    returns the text of the closest document
    """
    documents = [sentence]
    for name in sorted(os.listdir(corpus_path)):
        if not name.endswith('.md'):
            continue
        path = os.path.join(corpus_path, name)
        with open(path, 'r', encoding='utf-8') as handle:
            documents.append(handle.read())
    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')
    embeddings = model(documents)
    similarity = np.inner(embeddings, embeddings)
    closest = np.argmax(similarity[0, 1:])
    return documents[closest + 1]
