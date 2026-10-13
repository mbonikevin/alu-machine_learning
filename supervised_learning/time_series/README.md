# time_series

Forecasting the price of Bitcoin with an RNN. The model looks at the last 24
hours of BTC data and predicts the closing price of the next hour.

## Files

| File | What it does |
| --- | --- |
| `preprocess_data.py` | Cleans the raw coinbase and bitstamp files and saves `preprocessed.npz` |
| `forecast_btc.py` | Builds the `tf.data.Dataset`, trains the model and scores it |

## How to run it

Put the two raw csv files in this folder, then run:

```
./preprocess_data.py coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv
./forecast_btc.py
```

Running `preprocess_data.py` with no arguments uses those two file names by
default. It writes `preprocessed.npz` next to the scripts, and
`forecast_btc.py` reads that file.

## Choices made while preprocessing

- **Only data from 2017 on.** The early years have almost no volume and a
  price that says nothing about how BTC trades today.
- **Hours, not minutes.** The raw files have one row per minute. A single
  minute is mostly noise, so the minutes get rolled up into hours. Open is
  the first minute, close is the last one, high is the max, low is the min
  and the volumes are summed. That also turns a 1440 step window into a 24
  step one, which trains much faster.
- **Empty minutes get filled.** A minute with no trade shows up as NaN. The
  price carries over from the last minute that traded, and the volume for
  that minute is zero, because nothing was traded.
- **Fewer features.** Only close, high, low, volume in BTC and volume in USD
  are kept. Open and weighted price move almost exactly with close, so they
  add size without adding information.
- **Both exchanges.** Coinbase is used first and bitstamp fills the hours
  coinbase is missing.
- **Scaling.** Every feature is scaled with the mean and the standard
  deviation of the training part only. Using the whole set would leak the
  future into training.
- **Splitting by time.** 70% train, 20% validation, 10% test, cut in order.
  Shuffling before the split would let the model see the future.

## The model

A single LSTM layer with 64 units, dropout, then one dense output. The loss
is mean squared error and the optimizer is Adam. Early stopping watches the
validation loss and keeps the best weights.

The data reaches the model through a `tf.data.Dataset` that windows the
hours, so each sample is 24 hours of features and the label is the close of
the hour right after the window.
