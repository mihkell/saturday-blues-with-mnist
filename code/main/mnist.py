import os

import numpy as np
import pandas as pd
import tensorflow as tf

DATA_PATH = "../../data/"
PICKLE_FEATURES_FILENAME = "preprocessed_data.pkl"
PICKLE_LABEL_FILENAME = "preprocessed_labels.pkl"


def normalize(df):
    # AGE = 'Age'
    # ageMax = df[AGE].max()
    # ageMin = df[AGE].min()
    # df[AGE] = (df[AGE] - ageMin) / (ageMax - ageMin)
    # df[AGE] = (df[AGE] ** 0.8 * 2 - 1)
    df /= 255.0
    print(df.as_matrix().shape)
    df = np.reshape(df.as_matrix(), (df.shape[0], 28, 28, 1))
    print(df.shape[1:])

    return df


def clean_data(data_filename, do_not_use_pickle=False):
    features_file = DATA_PATH + PICKLE_FEATURES_FILENAME
    label_file = DATA_PATH + PICKLE_LABEL_FILENAME
    if do_not_use_pickle or (not os.path.exists(features_file) and not os.path.exists(label_file)):
        df = pd.read_csv(DATA_PATH + data_filename)

        input_data = df[["pixel" + str(i) for i in range(784)]]
        label_data = df["label"]

        input_data = normalize(input_data)

        pd.to_pickle(input_data, features_file)
        pd.to_pickle(label_data, label_file)
    return pd.read_pickle(features_file), pd.read_pickle(label_file)


def main():
    training_data, training_labels = clean_data("train.csv")

    filter1 = tf.Variable(tf.random_normal([7, 7, 1, 10]))
    filter2 = tf.Variable(tf.random_normal([2, 2, 10, 1]))

    input_ = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(tf.int32)

    strides = [1, 3, 3, 1]
    layer1 = tf.nn.conv2d(
        input_,
        filter1,
        padding='VALID',
        strides=strides)

    print(layer1)

    strides = [1, 1, 1, 1]
    layer2 = tf.nn.conv2d(
        layer1,
        filter2,
        padding='VALID',
        strides=strides)

    print(layer2)

    flat_image = tf.reshape(layer2, [-1, 49])
    out = tf.contrib.layers.fully_connected(flat_image, 10)

    print(out)

    hot = tf.one_hot(labels, 10, dtype=tf.float32)
    print(hot)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=hot))

    optimizer = tf.train.AdamOptimizer(0.002).minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 1):
            print(sess.run([out, cost, optimizer, hot], feed_dict={input_: training_data[:2],
                                                              labels: training_labels[:2]}))


if __name__ == '__main__':
    main()
