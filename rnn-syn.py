"""
Run rnn-syn experiments.
"""

import net
import tensorflow as tf
import numpy as np
import swdata

# Message analysis
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

random = np.random.RandomState(0)

# Task configuration
N_IMAGES = 3
MAX_SHAPES = 5

# Model configuration
N_HIDDEN = 256
N_COMM = 64
N_BATCH = 100


def build_feature_model(dataset, n_images, max_shapes, n_attrs):
    """
    Return an encoder-decoder model that uses the raw feature representation
    of ShapeWorld microworlds for communication. This is exactly the model used
    in Andreas and Klein (2017).
    """
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
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=t_labels, logits=t_pred))

    return t_features, t_labels, t_msg, t_pred, t_loss


def build_end2end_model(dataset, n_images, max_shapes, n_attrs):
    """
    Return an encoder-decoder model that uses the raw ShapeWorld image data for
    communication.
    """
    raise NotImplementedError


if __name__ == "__main__":
    #  tf.set_random_seed(0)

    N_TARGETS = 2
    N_DISTRACTORS = 1
    MAX_IMAGES = N_TARGETS + N_DISTRACTORS
    MAX_SHAPES = 2

    N_CONFIGS = 15
    SAMPLES_EACH_CONFIG = 200

    print("Generating data")
    train = []
    for n in range(N_CONFIGS):
        dataset = swdata.SpatialExtraSimple()
        print(dataset.relation, dataset.relation_dir)
        train.extend(
            dataset.generate(SAMPLES_EACH_CONFIG,
                             n_targets=N_TARGETS,
                             n_distractors=N_DISTRACTORS))

    n_attrs = len(train[0].worlds[0].shapes[0])

    print("Building model")
    t_features, t_labels, t_msg, t_pred, t_loss = build_feature_model(
        train, MAX_IMAGES, MAX_SHAPES, n_attrs)
    optimizer = tf.train.AdamOptimizer(0.001)
    o_train = optimizer.minimize(t_loss)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # ==== TRAIN ====
    acc_history = []
    loss_history = []

    print("Training")
    for epoch in range(10):
        loss = 0
        hits = 0
        total = 0
        for t in range(100):
            train_scenes = [train[random.randint(len(train))]
                            for _ in range(N_BATCH)]
            #  import ipdb; ipdb.set_trace()
            envs, labels = swdata.extract_envs_and_labels(
                train_scenes, MAX_IMAGES, MAX_SHAPES, n_attrs)
            l, preds, _ = session.run(
                    [t_loss, t_pred, o_train],
                    {t_features: envs, t_labels: labels})

            match = (preds > 0) == labels
            loss += l
            hits += np.all(match, axis=1).sum()
            total += len(match)

        acc = hits / total
        print("Epoch {}: Accuracy {}".format(epoch, acc))

        loss_history.append(loss)
        acc_history.append(acc)

    all_envs, all_labels = swdata.extract_envs_and_labels(
        train, MAX_IMAGES, MAX_SHAPES, n_attrs)
    all_msgs, all_preds = session.run(
            [t_msg, t_pred],
            {t_features: all_envs, t_labels: all_labels})
    msg_projs = PCA(2).fit_transform(all_msgs)
    sns.set_style('white')
    colors = None
    rels = list(map(lambda x: (x.relation, x.relation_dir), train))
    rels_unique = list(set(rels))
    rels_color_map = dict(zip(rels_unique,
                              ['red', 'green', 'blue', 'orange']))
    colors = [rels_color_map[rel] for rel in rels]

    plt.scatter(msg_projs[:, 0], msg_projs[:, 1], c=colors,
                s=25, linewidth=2)
    plt.show()
