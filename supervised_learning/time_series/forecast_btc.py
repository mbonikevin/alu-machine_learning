#!/usr/bin/env python3
"""train and validate an RNN that forecasts the next BTC close"""

import numpy as np
import tensorflow as tf


WINDOW = 24
BATCH = 256
EPOCHS = 20
AUTOTUNE = tf.data.experimental.AUTOTUNE


def make_dataset(data, close, window=WINDOW, batch=BATCH, shuffle=False):
    """turn a block of hours into (past 24 hours, next close) pairs

    data is the scaled feature matrix and close is the column that
    holds the closing price, the label is the close of the hour that
    comes right after the window
    """
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(window + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1, close]),
                num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10000)
    return ds.batch(batch).prefetch(AUTOTUNE)


def build_model(features, window=WINDOW):
    """stack an LSTM on top of the window and read one number off it"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(window, features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse',
                  metrics=['mae'])
    return model


def plot(history, actual, predicted, prefix='btc'):
    """save the loss curve and the test predictions as pictures"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is missing, skipping the plots')
        return
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend()
    plt.savefig(prefix + '_loss.png')
    plt.figure()
    plt.plot(actual[:500], label='actual')
    plt.plot(predicted[:500], label='predicted')
    plt.xlabel('hour')
    plt.ylabel('close in USD')
    plt.legend()
    plt.savefig(prefix + '_predictions.png')


def main():
    """load the preprocessed data, train the model and score it"""
    data = np.load('preprocessed.npz', allow_pickle=True)
    train, val, test = data['train'], data['val'], data['test']
    mean, std = data['mean'], data['std']
    close = int(data['close'])
    train_ds = make_dataset(train, close, shuffle=True)
    val_ds = make_dataset(val, close)
    test_ds = make_dataset(test, close)
    model = build_model(train.shape[1])
    model.summary()
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                            restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=[stop])
    mse, mae = model.evaluate(test_ds)
    print('test mse (scaled): {:.5f}'.format(mse))
    print('test mae in USD: {:.2f}'.format(mae * std[close]))
    predicted = model.predict(test_ds).reshape(-1)
    actual = np.concatenate([y.numpy() for _, y in test_ds])
    predicted = predicted * std[close] + mean[close]
    actual = actual * std[close] + mean[close]
    plot(history, actual, predicted)
    model.save('btc_forecast.h5')
    print('model saved to btc_forecast.h5')


if __name__ == '__main__':
    main()
