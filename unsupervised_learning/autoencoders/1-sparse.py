#!/usr/bin/env python3
"""this module creates a sparse autoencoder model"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """creates a sparse autoencoder and returns encoder, decoder and auto"""
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    reg = keras.regularizers.l1(lambtha)
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=reg)(x)
    encoder = keras.Model(inputs, latent)

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
