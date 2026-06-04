#!/usr/bin/env python3
"""this module creates a variational autoencoder model"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder and returns the three models"""
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """samples a point from the latent space"""
        mean, log_var = args
        shape = (keras.backend.shape(mean)[0], latent_dims)
        eps = keras.backend.random_normal(shape=shape)
        return mean + keras.backend.exp(log_var / 2) * eps

    z = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(inputs, [z, mean, log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    out = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, out)

    def vae_loss(y_true, y_pred):
        """computes the reconstruction plus kl divergence loss"""
        recon = keras.backend.binary_crossentropy(y_true, y_pred)
        recon = keras.backend.sum(recon, axis=1)
        kl = -0.5 * keras.backend.sum(
            1 + log_var - keras.backend.square(mean) -
            keras.backend.exp(log_var), axis=1)
        return recon + kl

    auto.compile(optimizer='adam', loss=vae_loss)
    return encoder, decoder, auto
