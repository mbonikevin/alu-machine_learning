#!/usr/bin/env python3
"""Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the instance

        Args:
            style_image: numpy.ndarray - the image used as a style reference
            content_image: numpy.ndarray - the image used as a content
                reference
            alpha: the weight for content cost
            beta: the weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
                style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or \
                content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so its pixels are in [0, 1] and its largest side
        is 512 pixels

        Args:
            image: numpy.ndarray of shape (h, w, 3) containing the image to
                be scaled

        Returns:
            the scaled image as a tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or \
                image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        h, w = image.shape[0], image.shape[1]
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image = image[tf.newaxis, ...]
        image = tf.image.resize_bicubic(image, (h_new, w_new))
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost

        The model is based on VGG19, its max pooling layers are replaced by
        average pooling layers and its outputs are the style layer outputs
        followed by the content layer output. It is saved in the instance
        attribute model.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.save('vgg_base_model')
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model('vgg_base_model',
                                         custom_objects=custom_objects)

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        model = tf.keras.models.Model(vgg.input, outputs)
        for layer in model.layers:
            layer.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer output

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)
                containing the layer output whose gram matrix should be
                calculated

        Returns:
            a tf.Tensor of shape (1, c, c) containing the gram matrix
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        h, w = int(input_layer.shape[1]), int(input_layer.shape[2])
        product = tf.einsum('bhwi,bhwj->bij', input_layer, input_layer)

        return product / tf.cast(h * w, tf.float32)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost

        Sets the instance attributes gram_style_features and content_feature.
        """
        vgg19 = tf.keras.applications.vgg19
        style_input = vgg19.preprocess_input(self.style_image * 255)
        content_input = vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(style_input)[:-1]
        content_output = self.model(content_input)[-1]

        self.gram_style_features = [self.gram_matrix(output)
                                    for output in style_outputs]
        self.content_feature = content_output
