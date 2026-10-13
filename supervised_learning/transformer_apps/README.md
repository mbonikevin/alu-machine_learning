# transformer_apps

Training a transformer to translate Portuguese into English, using the
`ted_hrlr_translate/pt_to_en` dataset.

| File | What is in it |
| --- | --- |
| `0-dataset.py` | `Dataset`, loads the splits and builds a subword tokenizer per language |
| `1-dataset.py` | adds `encode`, which wraps a sentence in start and end tokens |
| `2-dataset.py` | adds `tf_encode` so the encoding can run inside the graph |
| `3-dataset.py` | the full pipeline: filter, cache, shuffle, pad into batches, prefetch |
| `4-create_masks.py` | `create_masks`, the encoder mask, the combined mask and the decoder mask |
| `5-transformer.py` | the transformer itself, all of it in one file |
| `5-train.py` | `train_transformer`, the training loop |

How to run the training:

```
python3 -c "train_transformer = __import__('5-train').train_transformer; train_transformer(4, 128, 8, 512, 32, 2)"
```

Notes:

- The start of sentence token is the vocabulary size and the end of sentence
  token is that plus one, so the model vocabulary is the tokenizer size plus
  two.
- Padding is a zero, and both the loss and the masks ignore it.
- The learning rate climbs for the first 4000 steps and then falls off, which
  is the schedule from the attention paper.
