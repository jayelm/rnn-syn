"""
Run rnn-syn experiments.
"""

import net
import tensorflow as tf
import numpy as np
import swdata
from swdata import Scene, SWorld
import os
import sys
import time


random = np.random.RandomState()

assert Scene
assert SWorld


def build_feature_model(dataset,
                        n_images,
                        max_shapes,
                        n_attrs,
                        net_arch=(256, 64)):
    """
    Return an encoder-decoder model that uses the raw feature representation
    of ShapeWorld microworlds for communication. This is exactly the model used
    in Andreas and Klein (2017).
    """
    n_hidden, n_comm = net_arch

    # Each image represented as a max_shapes * n_attrs array
    n_image_features = max_shapes * n_attrs
    t_features = tf.placeholder(tf.float32, (None, n_images, n_image_features))

    # Whether an image is the target
    t_labels = tf.placeholder(tf.float32, (None, n_images))

    # Encoder observes both object features and target labels
    t_in = tf.concat((t_features, tf.expand_dims(t_labels, axis=2)), axis=2)

    cell = tf.contrib.rnn.GRUCell(n_hidden)
    with tf.variable_scope("enc1"):
        states1, hidden1 = tf.nn.dynamic_rnn(cell, t_in, dtype=tf.float32)
    t_hidden = hidden1
    t_msg = tf.nn.relu(net.linear(t_hidden, n_comm))

    # Decoder makes independent predictions for each set of object features
    t_expand_msg = tf.expand_dims(t_msg, axis=1)
    t_tile_message = tf.tile(t_expand_msg, (1, n_images, 1))
    t_out_feats = tf.concat((t_tile_message, t_features), axis=2)
    t_pred = tf.squeeze(
        net.mlp(t_out_feats, (n_hidden, 1), (tf.nn.relu, None)))
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


def gen_dataset(iargs):
    i, args = iargs
    if i is not None:
        t = time.time()
        print("{} Started".format(i))
        sys.stdout.flush()
    max_n, n_targets, n_distractors = args
    dataset = swdata.SpatialExtraSimple()
    train = dataset.generate(
        max_n, n_targets=n_targets, n_distractors=n_distractors)
    if i is not None:
        print("{} Finished ({}s)".format(i, round(time.time() - t, 2)))
        sys.stdout.flush()
    return train


def batches(train, batch_size, max_data=None):
    """
    Yield batches from `train`. Discards smallest batch sizes, like
    tf.train.Batch.
    """
    if max_data is not None:
        # Truncate list and yield normally
        yield from batches(train[:max_data], batch_size, max_data=None)
    else:
        for i in range(0, len(train), batch_size):
            batch = train[i:i + batch_size]
            if len(batch) == batch_size:
                yield batch


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='rnn-syn', formatter_class=ArgumentDefaultsHelpFormatter)

    data_opts = parser.add_argument_group('data', 'options for data gen')
    data_opts.add_argument(
        '--data',
        type=str,
        required=True,
        help='Folder of dataset to load')

    net_opts = parser.add_argument_group('net', 'options for net architecture')
    net_opts.add_argument(
        '--n_hidden', type=int, default=256, help='GRUCell hidden layer size')
    net_opts.add_argument(
        '--n_comm', type=int, default=64, help='Communication layer size')
    net_opts.add_argument(
        '--comm_type',
        type=str,
        default='continuous',
        choices=['continuous', 'discrete'],
        help='Communication channel type')

    train_opts = parser.add_argument_group('train', 'options for net training')
    train_opts.add_argument(
        '--restore', action='store_true', help='Restore model')
    train_opts.add_argument(
        '--restore_path',
        type=str,
        default='saves/rnn-syn-model',
        help='Restore filepath')
    train_opts.add_argument(
        '--save',
        action='store_true',
        help='Save model file')
    train_opts.add_argument(
        '--save_path',
        type=str,
        default='saves/rnn-syn-model',
        help='Save model filepath')
    train_opts.add_argument(
        '--tf_seed', type=int, default=None, help='Random TensorFlow seed')
    train_opts.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    train_opts.add_argument(
        '--epochs', type=int, default=16, help='Number of training epochs')
    train_opts.add_argument(
        '--max_data',
        type=int,
        default=None,
        help='Max size of training data (rest discarded)')

    test_opts = parser.add_argument_group('train', 'options for net testing')
    test_opts.add_argument('--test', action='store_true',
                           help='do testing')
    test_opts.add_argument('--test_split', type=float, default=0.2,
                           help='% of dataset to test on')
    test_opts.add_argument('--test_no_unique', action='store_true',
                           help='Don\'t require testing unique configs')

    save_opts = parser.add_argument_group('save messages')
    save_opts.add_argument('--no_save_msgs', action='store_true',
                           help='Don\'t save comptued messages after testing')
    save_opts.add_argument('--msgs_file', default='{data}-msgs.npz',
                           help='Save location (can use parser options)')

    args = parser.parse_args()

    if args.comm_type == 'discrete':
        raise NotImplementedError

    if args.tf_seed is not None:
        tf.set_random_seed(args.tf_seed)

    print("Loading data")
    train, metadata = swdata.load_scenes(args.data, gz=True)

    # Do a split
    if args.test:
        # Keep unique configs only
        if not args.test_no_unique:
            # TODO: Support different kinds of testing (e.g. left/right)
            unique_sets = []
            seen_configs = set()
            for config_data, config_md in zip(train, metadata['configs']):
                config_hashable = (tuple(config_md['distractor']),
                                   tuple(config_md['target']),
                                   config_md['relation'],
                                   config_md['relation_dir'])
                if config_hashable not in seen_configs:
                    seen_configs.add(config_hashable)
                    unique_sets.append(config_data)
            random.shuffle(unique_sets)
            train, test = swdata.train_test_split(unique_sets,
                                                  test_split=args.test_split)
            train = swdata.flatten(train)
            test = swdata.flatten(test)
            random.shuffle(train)
            random.shuffle(test)
        else:
            train, test = swdata.train_test_split(train,
                                                  test_split=args.test_split)
            train = swdata.flatten(train)
            random.shuffle(train)
        print("Train:", len(train), "Test:", len(test))
    else:
        # Just train on everything
        train = swdata.flatten(train)

    max_images = metadata['n_targets'] + metadata['n_distractors']
    max_shapes = 2
    n_attrs = len(train[0].worlds[0].shapes[0])

    # Throw out duplicate configs?

    print("Building model")
    t_features, t_labels, t_msg, t_pred, t_loss = build_feature_model(
        train,
        max_images,
        max_shapes,
        n_attrs,
        net_arch=(args.n_hidden, args.n_comm))
    optimizer = tf.train.AdamOptimizer(0.001)
    o_train = optimizer.minimize(t_loss)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # ==== TRAIN ====
    if args.restore:
        saver = tf.train.Saver()
        saver.restore(session, args.restore_path)
    else:
        acc_history = []
        loss_history = []

        print("Training")
        for epoch in range(args.epochs):
            loss = 0
            hits = 0
            total = 0
            # Shuffle training data
            random.shuffle(train)
            for batch in batches(
                    train, args.batch_size, max_data=args.max_data):
                envs, labels = swdata.extract_envs_and_labels(
                    batch, max_images, max_shapes, n_attrs)
                l, preds, _ = session.run([t_loss, t_pred, o_train], {
                    t_features: envs,
                    t_labels: labels
                })

                match = (preds > 0) == labels
                loss += l
                hits += np.all(match, axis=1).sum()
                total += len(match)

            acc = hits / total
            print("Epoch {}: Accuracy {}, Loss {}".format(epoch, acc, loss))

            loss_history.append(loss)
            acc_history.append(acc)

        if args.save:
            saver = tf.train.Saver()
            saver.save(session, args.save_path)

    # ==== TEST ====
    test_or_train = test if args.test else train
    all_envs, all_labels = swdata.extract_envs_and_labels(
        test_or_train, max_images, max_shapes, n_attrs)
    all_msgs, all_preds = session.run([t_msg, t_pred], {
        t_features: all_envs, t_labels: all_labels})

    if args.test:  # Print test accuracy
        match = (all_preds > 0) == all_labels
        print("Test accuracy: {}".format(np.all(match, axis=1).sum() / len(match)))

    relations = np.array([x.relation[0] for x in test_or_train])
    relation_dirs = np.array([x.relation_dir for x in test_or_train])
    if not args.no_save_msgs:
        print("Saving model predictions")
        np.savez(args.msgs_file.format(**vars(args)),
                 msgs=all_msgs,
                 preds=all_preds,
                 obs=all_labels,
                 relations=relations,
                 relation_dirs=relation_dirs)
