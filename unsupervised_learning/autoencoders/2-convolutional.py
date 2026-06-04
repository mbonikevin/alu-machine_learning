#!/usr/bin/env python3
"""this module creates a convolutional autoencoder model"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder and returns the three models"""
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs, x)

    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[0], (3, 3), padding='valid',
                            activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), padding='same',
                                  activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
