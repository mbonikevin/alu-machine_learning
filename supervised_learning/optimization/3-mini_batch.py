#!/usr/bin/env python3
"""this module trains a loaded model using mini-batch gradient descent"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """returns the path where the model was saved"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]
        if m % batch_size == 0:
            steps = m // batch_size
        else:
            steps = m // batch_size + 1

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
        return saver.save(sess, save_path)
