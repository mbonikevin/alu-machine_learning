#!/usr/bin/env python3
"""this module builds, trains and saves a full optimized model"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_layer(prev, n, activation):
    """creates a plain dense layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)
    gamma = tf.Variable(tf.ones([1, n]), name='gamma')
    beta = tf.Variable(tf.zeros([1, n]), name='beta')
    mean, var = tf.nn.moments(z, axes=[0])
    z_norm = tf.nn.batch_normalization(z, mean, var, beta, gamma, 1e-8)
    return activation(z_norm)


def forward_prop(x, layers, activations):
    """builds the forward propagation graph"""
    a = x
    for i in range(len(layers)):
        if i == len(layers) - 1:
            a = create_layer(a, layers[i], activations[i])
        else:
            a = create_batch_norm_layer(a, layers[i], activations[i])
    return a


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """returns the path where the model was saved"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha_op = tf.train.inverse_time_decay(alpha, global_step, 1,
                                           decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(alpha_op, beta1, beta2,
                                      epsilon).minimize(loss)
    tf.add_to_collection('train_op', train_op)
    inc = tf.assign_add(global_step, 1)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    m = X_train.shape[0]
    if m % batch_size == 0:
        steps = m // batch_size
    else:
        steps = m // batch_size + 1

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs + 1):
            tc, ta = sess.run([loss, accuracy],
                              feed_dict={x: X_train, y: Y_train})
            vc, va = sess.run([loss, accuracy],
                              feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(tc))
            print("\tTraining Accuracy: {}".format(ta))
            print("\tValidation Cost: {}".format(vc))
            print("\tValidation Accuracy: {}".format(va))
            if epoch < epochs:
                X_s, Y_s = shuffle_data(X_train, Y_train)
                for step in range(steps):
                    start = step * batch_size
                    end = start + batch_size
                    X_b = X_s[start:end]
                    Y_b = Y_s[start:end]
                    sess.run(train_op, feed_dict={x: X_b, y: Y_b})
                    if (step + 1) % 100 == 0:
                        sc, sa = sess.run([loss, accuracy],
                                          feed_dict={x: X_b, y: Y_b})
                        print("\tStep {}:".format(step + 1))
                        print("\t\tCost: {}".format(sc))
                        print("\t\tAccuracy: {}".format(sa))
                sess.run(inc)
        return saver.save(sess, save_path)
