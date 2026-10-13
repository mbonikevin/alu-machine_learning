# nlp_metrics

Scoring a translation against reference translations with BLEU, written with
numpy only. No nltk.

| File | What it does |
| --- | --- |
| `0-uni_bleu.py` | `uni_bleu(references, sentence)`, BLEU on single words |
| `1-ngram_bleu.py` | `ngram_bleu(references, sentence, n)`, BLEU on n-grams |
| `2-cumulative_bleu.py` | `cumulative_bleu(references, sentence, n)`, all sizes from 1 to n, weighted evenly |

How the score is built:

- Count how many of the sentence n-grams show up in a reference. A word that
  the sentence repeats more often than any reference does only counts as many
  times as the best reference has it.
- Divide by how many n-grams the sentence has. That is the precision.
- Short sentences get punished, because a one word sentence would otherwise
  score perfectly. If the sentence is shorter than the closest reference, the
  precision is multiplied by `exp(1 - r / c)`, where `r` is the length of the
  reference closest to the sentence and `c` is the sentence length.
- For the cumulative score, the precisions of every n-gram size are combined
  with a geometric mean before the penalty is applied.
