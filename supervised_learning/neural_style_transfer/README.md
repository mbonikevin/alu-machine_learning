# Neural Style Transfer

Takes a content image and a style image and makes a new image that keeps the
content of the first one but looks like it was painted in the style of the
second one.

The `NST` class is built up task by task. Each file adds one more method on
top of the previous one.

## Files

| File | What it adds |
| --- | --- |
| `0-neural_style.py` | the class, input checks and `scale_image` |
| `1-neural_style.py` | `load_model`, a VGG19 with average pooling |
| `2-neural_style.py` | `gram_matrix` |
| `3-neural_style.py` | `generate_features` |
| `4-neural_style.py` | `layer_style_cost` |
| `5-neural_style.py` | `style_cost` |
| `6-neural_style.py` | `content_cost` |
| `7-neural_style.py` | `total_cost` |
| `8-neural_style.py` | `compute_grads` |
| `9-neural_style.py` | `generate_image` |
| `10-neural_style.py` | `variational_cost` and the `var` weight |

## How it works

The content and style images are scaled so the biggest side is 512 pixels and
the pixel values sit between 0 and 1.

A VGG19 model is loaded without its top layers and its max pooling layers are
swapped for average pooling. The model outputs the five style layers plus the
content layer. Nothing in it is trainable.

Style is measured with gram matrices. For every style layer we compare the
gram matrix of the generated image to the one of the style image. Content is
measured by comparing the content layer output of the generated image to the
one of the content image.

The total cost is `alpha * content + beta * style`, and in task 10 we also add
`var * variational`, which smooths out the noise.

The generated image starts as a copy of the content image. Adam gradient
descent then moves the pixels themselves, not the weights, and the image with
the lowest cost is the one we keep.

## Requirements

- Ubuntu 16.04 LTS, `python3` (3.5)
- `numpy` 1.15, `tensorflow` 1.12 (eager execution)
- `pycodestyle` 2.4

## Usage

Put `golden_gate.jpg` and `starry_night.jpg` in this folder, then run a main
file:

```
./9-main.py
```
