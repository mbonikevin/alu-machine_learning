#!/usr/bin/env python3
"""training of a transformer for portuguese to english"""

import tensorflow.compat.v2 as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """the warmup learning rate from the attention paper"""

    def __init__(self, dm, warmup_steps=4000):
        """keep the model depth and the length of the warmup"""
        super(Schedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """climb for the first steps, then fall off slowly"""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, epochs):
    """create and train a transformer on the ted talk dataset

    N is the number of blocks in the encoder and the decoder
    dm is the model depth
    h is the number of heads
    hidden is the size of the hidden feed forward layer
    max_len is the longest sentence to keep, in tokens
    epochs is how many passes over the training data to take
    returns the trained model
    """
    data = Dataset(32, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2
    model = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                        max_len, max_len)
    optimizer = tf.keras.optimizers.Adam(Schedule(dm), beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    def loss_function(real, pred):
        """cross entropy that ignores whatever is padding"""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = loss_object(real, pred)
        loss *= tf.cast(mask, loss.dtype)
        return tf.reduce_mean(loss)

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for batch, (inputs, target) in enumerate(data.data_train):
            target_input = target[:, :-1]
            target_real = target[:, 1:]
            masks = create_masks(inputs, target_input)
            with tf.GradientTape() as tape:
                predictions = model(inputs, target_input, True, masks[0],
                                    masks[1], masks[2])
                loss = loss_function(target_real, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          model.trainable_variables))
            train_loss(loss)
            train_accuracy(target_real, predictions)
            if batch % 50 == 0:
                print('Epoch {}, batch {}: loss {} accuracy {}'.format(
                    epoch + 1, batch, train_loss.result(),
                    train_accuracy.result()))
        print('Epoch {}: loss {} accuracy {}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))
    return model
