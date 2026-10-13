# qa_bot

A bot that answers questions by pulling the answer straight out of a
document, using BERT fine tuned on SQuAD.

| File | What is in it |
| --- | --- |
| `0-qa.py` | `question_answer(question, reference)`, finds the answer inside one document |
| `1-loop.py` | the chat loop on its own, it asks and answers nothing yet |
| `2-qa.py` | `answer_loop(reference)`, the loop plus the BERT answers |
| `3-semantic_search.py` | `semantic_search(corpus_path, sentence)`, picks the closest document |
| `4-qa.py` | `question_answer(corpus_path)`, searches the corpus then answers |

How it works:

- The question and the document are tokenized together as
  `[CLS] question [SEP] document [SEP]`. The model gives a score to every
  token for being the start of the answer and another for being the end, and
  the answer is the stretch between the best two.
- Position 0 is skipped when looking for the start and the end, because that
  is the `[CLS]` token, which is how the model says it found nothing.
- An empty answer turns into `Sorry, I do not understand your question.`
- `exit`, `quit`, `goodbye` and `bye` end the chat, in any casing, and the
  bot says `Goodbye`.
- The search uses the universal sentence encoder. Every document and the
  question get one vector each, and the document whose vector points most in
  the same direction as the question wins.

Needs:

```
pip install --user tensorflow-hub transformers
```
