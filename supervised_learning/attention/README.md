# attention

Attention for machine translation, then the whole transformer built up piece
by piece.

| File | What is in it |
| --- | --- |
| `0-rnn_encoder.py` | `RNNEncoder`, an embedding and a GRU |
| `1-self_attention.py` | `SelfAttention`, the context vector for the decoder |
| `2-rnn_decoder.py` | `RNNDecoder`, one decoding step with attention |
| `4-positional_encoding.py` | `positional_encoding`, the sine and cosine position vectors |
| `5-sdp_attention.py` | `sdp_attention`, scaled dot product attention |
| `6-multihead_attention.py` | `MultiHeadAttention`, attention run in parallel heads |
| `7-transformer_encoder_block.py` | `EncoderBlock` |
| `8-transformer_decoder_block.py` | `DecoderBlock` |
| `9-transformer_encoder.py` | `Encoder`, a stack of encoder blocks |
| `10-transformer_decoder.py` | `Decoder`, a stack of decoder blocks |
| `11-transformer.py` | `Transformer`, the full network |

Needs tensorflow 1.15:

```
pip install --user tensorflow==1.15
```
