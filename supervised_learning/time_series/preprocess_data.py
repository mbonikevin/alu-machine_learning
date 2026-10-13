#!/usr/bin/env python3
"""preprocessing of the raw BTC minute data for forecasting"""

import sys
import numpy as np
import pandas as pd


PRICES = ['Open', 'High', 'Low', 'Close', 'Weighted_Price']
VOLUMES = ['Volume_(BTC)', 'Volume_(Currency)']
FEATURES = ['Close', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)']
START = '2017-01-01'
HOURLY = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
          'Volume_(BTC)': 'sum', 'Volume_(Currency)': 'sum',
          'Weighted_Price': 'mean'}


def load(path):
    """read one raw csv and index it by its unix timestamp

    only the rows from 2017 on are kept, the years before that have
    little volume and a price that no longer says anything useful
    """
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp').sort_index()
    return df[df.index >= START]


def to_hourly(df):
    """fill the empty minutes and roll the minutes up into hours

    a minute with no trade comes back as NaN, the price is carried
    over from the last minute that traded and the volume is a zero
    """
    df = df.copy()
    df[PRICES] = df[PRICES].ffill()
    df[VOLUMES] = df[VOLUMES].fillna(0)
    df = df.dropna()
    return df.resample('H').agg(HOURLY).dropna()


def merge(coinbase, bitstamp):
    """keep coinbase and use bitstamp only for the hours it is missing"""
    return coinbase.combine_first(bitstamp).dropna()


def split(data, train_frac=0.7, val_frac=0.2):
    """cut the series in three along time, never at random

    shuffling first would let the model peek at the future
    """
    n = data.shape[0]
    end_train = int(n * train_frac)
    end_val = int(n * (train_frac + val_frac))
    return data[:end_train], data[end_train:end_val], data[end_val:]


def scale(train, val, test):
    """put every feature on the same scale with the training stats"""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, \
        mean, std


def preprocess(coinbase_path, bitstamp_path, out_path='preprocessed.npz'):
    """build the hourly dataset and write it next to the scripts"""
    coinbase = to_hourly(load(coinbase_path))
    bitstamp = to_hourly(load(bitstamp_path))
    hourly = merge(coinbase, bitstamp)[FEATURES]
    data = hourly.values.astype('float32')
    train, val, test = split(data)
    train, val, test, mean, std = scale(train, val, test)
    np.savez(out_path, train=train, val=val, test=test, mean=mean,
             std=std, features=np.array(FEATURES),
             close=np.array(FEATURES.index('Close')))
    print('hours: {}, features: {}'.format(data.shape[0], data.shape[1]))
    print('train: {}, val: {}, test: {}'.format(train.shape[0],
                                                val.shape[0], test.shape[0]))
    print('saved to {}'.format(out_path))
    return train, val, test


def main():
    """run the preprocessing from the command line"""
    coinbase = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    bitstamp = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    if len(sys.argv) > 2:
        coinbase, bitstamp = sys.argv[1], sys.argv[2]
    preprocess(coinbase, bitstamp)


if __name__ == '__main__':
    main()
