# word_embeddings

Different ways of turning words into vectors.

| File | What it does |
| --- | --- |
| `0-bag_of_words.py` | Counts how many times each word shows up in each sentence |
| `1-tf_idf.py` | Same idea, but a word that shows up everywhere counts for less |
| `2-word2vec.py` | Trains a gensim word2vec model, CBOW or skip-gram |
| `3-gensim_to_keras.py` | Hands a trained word2vec model to keras as an Embedding layer |
| `4-fasttext.py` | Trains a gensim fasttext model, which also looks at parts of words |
| `5-elmo` | The answer to the ELMo question |

Needs `gensim` 3.8 and `keras` 2.2.5:

```
pip install --user gensim==3.8
pip install --user keras==2.2.5
```
