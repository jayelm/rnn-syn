"""
Run rnn-syn experiments.
"""

import net
import tensorflow as tf
import numpy as np
import swdata
import multiprocessing as mp
import os

# Message analysis
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

random = np.random.RandomState(0)


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


def gen_dataset(args):
    max_n, n_targets, n_distractors = args
    dataset = swdata.SpatialExtraSimple()
    return dataset.generate(
        max_n, n_targets=n_targets, n_distractors=n_distractors)


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
    import sys

    parser = ArgumentParser(
        description='rnn-syn', formatter_class=ArgumentDefaultsHelpFormatter)

    data_opts = parser.add_argument_group('data', 'options for data gen')

    data_opts.add_argument(
        '--load_data',
        type=str,
        default=None,
        help='Folder of dataset to load')
    data_opts.add_argument(
        '--n_configs',
        type=int,
        default=15,
        help='Number of random scene configs to sample')
    data_opts.add_argument(
        '--samples_each_config',
        type=int,
        default=100,
        help='(max) number of scenes to sample per config')
    data_opts.add_argument(
        '--n_targets', type=int, default=2, help='Number of targets per scene')
    data_opts.add_argument(
        '--n_distractors',
        type=int,
        default=1,
        help='Number of distractors per scene')
    data_opts.add_argument(
        '--n_cpu',
        type=int,
        default=1,
        help='Number of cpus to use for mp (1 disables mp)')

    net_opts = parser.add_argument_group('net', 'options for net training')
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

    train_opts = parser.add_argument_group('train', 'Options for net training')
    do_training = parser.add_mutually_exclusive_group()
    do_training.add_argument(
        '--no_train', action='store_true', help='Don\'t train')
    do_training.add_argument(
        '--restore', action='store_true', help='Restore model')
    do_training.add_argument(
        '--restore_path',
        type=str,
        default='saves/rnn-syn-model',
        help='Restore filepath')
    train_opts.add_argument(
        '--save_path',
        type=str,
        default='saves/rnn-syn-model',
        help='Save model file')
    train_opts.add_argument(
        '--tf_seed', type=int, default=None, help='Random TensorFlow seed')
    train_opts.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    train_opts.add_argument(
        '--epochs', type=int, default=128, help='Number of training epochs')
    train_opts.add_argument(
        '--max_data',
        type=int,
        default=None,
        help='Max size of training data (rest discarded)')

    args = parser.parse_args()

    if args.comm_type == 'discrete':
        raise NotImplementedError

    if args.tf_seed is not None:
        tf.set_random_seed(args.tf_seed)

    max_images = args.n_targets + args.n_distractors
    max_shapes = 2

    if not args.restore and args.load_data is None:
        print("Generating data")
        dataset_args = [(args.samples_each_config, args.n_targets,
                         args.n_distractors) for _ in range(args.n_configs)]
        if args.n_cpu == 1:  # Non-mp
            dataset_iter = map(gen_dataset, dataset_args)
        else:
            pool = mp.Pool(args.n_cpu)
            dataset_iter = pool.map(gen_dataset, dataset_args)
            pool.close()
            pool.join()

        train = []
        for train_subset in dataset_iter:
            train.extend(train_subset)

        # Save data
        save_file = '{}-{}-{}-{}t-{}d.pkl.gz'.format(
            len(train), args.n_configs, args.samples_each_config,
            args.n_targets, args.n_distractors)
        swdata.pickle_scenes(
            train, save_file=os.path.join('data', save_file), gz=True)
    else:
        if not args.load_data:
            assert args.restore
            raise RuntimeError("Must supply dataset if restoring model")
        print("Loading data")
        train = swdata.load_scenes(args.load_data, gz=True)

    n_attrs = len(train[0].worlds[0].shapes[0])

    if args.no_train:
        sys.exit(0)

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
            print("Epoch {}: Accuracy {}".format(epoch, acc))

            loss_history.append(loss)
            acc_history.append(acc)

        saver = tf.train.Saver()
        saver.save(session, args.save_path)

    all_envs, all_labels = swdata.extract_envs_and_labels(
        train, max_images, max_shapes, n_attrs)
    all_msgs, all_preds = session.run([t_msg, t_pred], {
        t_features: all_envs,
        t_labels: all_labels
    })
    msg_projs = PCA(2).fit_transform(all_msgs)
    sns.set_style('white')
    colors = None
    rels = list(map(lambda x: (x.relation, x.relation_dir), train))
    rels_unique = list(set(rels))
    rels_color_map = dict(zip(rels_unique, ['red', 'green', 'blue', 'orange']))
    colors = [rels_color_map[rel] for rel in rels]

    plt.scatter(msg_projs[:, 0], msg_projs[:, 1], c=colors, s=25, linewidth=2)
    plt.show()
