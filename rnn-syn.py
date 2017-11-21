"""
Run rnn-syn experiments.
"""

import net
import tensorflow as tf
import numpy as np
import swdata

random = np.random.RandomState(0)

# Task configuration
N_IMAGES = 2
MAX_SHAPES = 5

# Model configuration
N_HIDDEN = 256
N_COMM = 64
N_BATCH = 100


def build_model(dataset, n_images, max_shapes, n_attrs):
    # Each image represented as a max_shapes * n_attrs array
    n_image_features = max_shapes * n_attrs
    t_features = tf.placeholder(tf.float32,
                                (None, n_images, n_image_features))

    # Whether an image is the target
    t_labels = tf.placeholder(tf.float32, (None, n_images))

    # Encoder observes both object features and target labels
    t_in = tf.concat((t_features, tf.expand_dims(t_labels, axis=2)), axis=2)

    cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
    with tf.variable_scope("enc1"):
        states1, hidden1 = tf.nn.dynamic_rnn(cell, t_in, dtype=tf.float32)
    t_hidden = hidden1
    t_msg = tf.nn.relu(net.linear(t_hidden, N_COMM))

    # Decoder makes independent predictions for each set of object features
    t_expand_msg = tf.expand_dims(t_msg, axis=1)
    t_tile_message = tf.tile(t_expand_msg, (1, n_images, 1))
    t_out_feats = tf.concat((t_tile_message, t_features), axis=2)
    t_pred = tf.squeeze(
        net.mlp(t_out_feats, (N_HIDDEN, 1), (tf.nn.relu, None)))
    t_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=t_labels, logits=t_pred))

    return t_features, t_labels, t_msg, t_pred, t_loss


if __name__ == "__main__":
    tf.set_random_seed(0)

    train = swdata.gen_example_data(10000, n_images=N_IMAGES,
                                    max_shapes=MAX_SHAPES, target='simple')
    train = swdata.flatten_scenes(train)
    n_attrs = len(train[0].images[0][0])

    t_features, t_labels, t_msg, t_pred, t_loss = build_model(
        train, N_IMAGES, MAX_SHAPES, n_attrs)
    optimizer = tf.train.AdamOptimizer(0.001)
    o_train = optimizer.minimize(t_loss)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # ==== TRAIN ====
    acc_history = []
    loss_history = []

    for epoch in range(10):
        loss = 0
        hits = 0
        total = 0
        for t in range(100):
            train_scenes = [train[random.randint(len(train))]
                            for _ in range(N_BATCH)]
            envs, labels = swdata.extract_envs_and_labels(
                train_scenes, N_IMAGES, MAX_SHAPES, n_attrs)
            l, preds, _ = session.run(
                    [t_loss, t_pred, o_train],
                    {t_features: envs, t_labels: labels})
            binary_labels = np.where(labels)[1]
            binary_preds = np.argmax(preds, axis=1)
            loss += l
            hits += sum(binary_labels == binary_preds)
            total += len(binary_labels)

        acc = hits / total
        print("Epoch {}: Accuracy {}".format(epoch, acc))

        loss_history.append(loss)
        acc_history.append(acc)
