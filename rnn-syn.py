"""
Run rnn-syn experiments.

TODO: Maybe variation in number of images is a big confounding factor in
messages...could just keep that constant!
"""

import net
import tensorflow as tf
import numpy as np
import swdata
from swdata import (AsymScene, Scene, SWorld, TrainEx, load_components,
                    make_from_components)
import sys
import os
from tensorflow.python import debug as tf_debug
import pandas as pd
import itertools
import gc
from scipy.misc import imsave

RNN_CELLS = {
    'gru': tf.contrib.rnn.GRUCell,
    'lstm': tf.contrib.rnn.LSTMCell,
}

assert AsymScene
assert Scene
assert SWorld
assert TrainEx


def mkconfig(a, b, n=1000):
    # Just sorts them in order so we can reliably identify the config.
    return '{}-{}-{}'.format(n, *sorted([a, b]))


def mkconfigs(arr, n=1000):
    return [mkconfig(a, b, n=n) for a, b in itertools.combinations(arr, 2)]


CONFIGS = {
    # Generalization to new color/shape pair (triangle + red)
    # After seeing 2 colors and shapes
    'shape_color_generalization_1': {
        'train': [
            mkconfig('square-blue', 'square-red'),
            mkconfig('square-blue', 'triangle-blue'),
            mkconfig('square-red', 'triangle-blue')
        ],
        'test': [
            mkconfig('square-blue', 'triangle-red'),
            mkconfig('square-red', 'triangle-red'),
            mkconfig('triangle-blue', 'triangle-red')
        ]
    },
    # Generalization to new color/shape pair
    # After seeing 3 colors and 2 shapes
    'shape_color_generalization_2': {
        'train':
        mkconfigs([
            'square-blue', 'square-red', 'triangle-blue', 'square-green',
            'triangle-green'
        ]),
        'test': [
            mkconfig('triangle-red', b) for b in [
                'square-blue', 'square-red', 'triangle-blue', 'square-green',
                'triangle-green'
            ]
        ]
    },
    # After seeing 3 colors and 3 shapes, importantly: trained on red
    'shape_color_generalization_3': {
        'train':
        mkconfigs([
            'square-red', 'square-blue', 'square-green',
            'triangle-blue', 'triangle-green',
            'circle-red', 'circle-blue', 'circle-green'
        ]),
        'test': [
            mkconfig('triangle-red', b) for b in [
                'square-red', 'square-blue', 'square-green',
                'triangle-blue', 'triangle-green',
                'circle-red', 'circle-blue', 'circle-green'
            ]
        ]
    },
    # Generalization to new pair (does it with 100% accuracy, meaning messages encode target/referent
    'new_pair_generalization_1': {
        'train': [
            mkconfig('square-blue', 'square-red'),
            mkconfig('square-red', 'triangle-blue')
        ],
        'test': [mkconfig('square-blue', 'triangle-blue')]
    }
}


def find_true_example(envslabels):
    envs, labels = envslabels
    for env, label in zip(envs, labels):
        if label == 1.0:
            return env
    raise RuntimeError("Coudln't find a True label")


def build_feature_model(n_images,
                        max_shapes,
                        n_attrs,
                        net_arch=(256, 64),
                        discrete=False,
                        rnncell=tf.contrib.rnn.GRUCell,
                        asym=False):
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

    if asym:
        # Listener sees separate labels and features
        t_features_l = tf.placeholder(tf.float32,
                                      (None, n_images, n_image_features))
        t_labels_l = tf.placeholder(tf.float32, (None, n_images))

    # Encoder observes both object features and target labels
    t_labels_exp = tf.expand_dims(t_labels, axis=2)
    t_in = tf.concat((t_features, t_labels_exp), axis=2)

    if rnncell == tf.contrib.rnn.LSTMCell:
        cell = rnncell(n_hidden, state_is_tuple=False)
    else:
        cell = rnncell(n_hidden)
    with tf.variable_scope("enc1"):
        states1, hidden1 = tf.nn.dynamic_rnn(cell, t_in, dtype=tf.float32)
    t_hidden = hidden1
    t_msg = tf.nn.relu(net.linear(t_hidden, n_comm, 'linear_speaker'))
    if discrete:
        t_msg_discrete = tf.one_hot(
            tf.argmax(t_msg, axis=1), depth=n_comm, name='discretize')

    # Decoder makes independent predictions for each set of object features
    t_expand_msg = tf.expand_dims(
        t_msg_discrete if discrete else t_msg, axis=1)
    t_tile_message = tf.tile(t_expand_msg, (1, n_images, 1))

    if asym:  # Encode listener features
        t_out_feats = tf.concat((t_tile_message, t_features_l), axis=2)
    else:
        t_out_feats = tf.concat((t_tile_message, t_features), axis=2)

    t_pred = tf.squeeze(
        net.mlp(t_out_feats, (n_hidden, 1), (tf.nn.relu, None)))

    if asym:  # Loss wrt listener labels
        t_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=t_labels_l, logits=t_pred))
        return (t_features, t_labels, t_features_l, t_labels_l,
                (t_msg_discrete if discrete else t_msg), t_pred, t_loss)
    else:
        t_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=t_labels, logits=t_pred))
        return (t_features, t_labels, (t_msg_discrete
                                       if discrete else t_msg), t_pred, t_loss)


def build_end2end_model(n_images,
                        image_dim=(64, 64, 3),
                        net_arch=(256, 64, 1024),
                        discrete=False,
                        rnncell=tf.contrib.rnn.GRUCell,
                        asym=False):
    """
    Return an encoder-decoder model that uses raw ShapeWorld images

    net_arch:
        (number of GRU hidden units, message dimensionality,
         convnet toplevel layer dimensionality)

    discrete:
        Discretize by one-hot encoding the message. Then message
        dimensionality arg of net_arch encodes vocabulary size
    """
    n_hidden, n_comm, n_toplevel_conv = net_arch

    # The raw image representation, of shape n_images * image_dim
    t_features_raw = tf.placeholder(
        tf.float32, (None, n_images) + image_dim, name='features_speaker')

    t_features_toplevel_enc = net.convolve(t_features_raw, n_images,
                                           n_toplevel_conv, 'conv_speaker')

    # Whether an image is the target
    t_labels = tf.placeholder(
        tf.float32, (None, n_images), name='labels_speaker')

    if asym:
        # Listener observes own features/labels
        t_features_raw_l = tf.placeholder(
            tf.float32, (None, n_images) + image_dim, name='features_listener')
        t_labels_l = tf.placeholder(
            tf.float32, (None, n_images), name='labels_listener')

    # Encoder observes both object features and target labels
    t_labels_exp = tf.expand_dims(t_labels, axis=2)
    t_in = tf.concat(
        (t_features_toplevel_enc, t_labels_exp), axis=2, name='input_speaker')

    if rnncell == tf.contrib.rnn.LSTMCell:
        cell = rnncell(n_hidden, state_is_tuple=False)
    else:
        cell = rnncell(n_hidden)
    with tf.variable_scope("enc1"):
        states1, hidden1 = tf.nn.dynamic_rnn(cell, t_in, dtype=tf.float32)
    t_hidden = hidden1
    t_msg = tf.nn.relu(
        net.linear(t_hidden, n_comm, 'linear_speaker'), name='message')

    if discrete:
        t_msg_discrete = tf.one_hot(
            tf.argmax(t_msg, axis=1), depth=n_comm, name='message_discrete')

    # Decoder makes independent predictions for each set of object features
    with tf.name_scope('message_process'):
        t_expand_msg = tf.expand_dims(
            t_msg_discrete if discrete else t_msg, axis=1)
        t_tile_message = tf.tile(t_expand_msg, (1, n_images, 1))

    if asym:
        t_features_toplevel_dec = net.convolve(
            t_features_raw_l, n_images, n_toplevel_conv, 'conv_listener')
    else:
        t_features_toplevel_dec = net.convolve(
            t_features_raw, n_images, n_toplevel_conv, 'conv_listener')
    t_out_feats = tf.concat(
        (t_tile_message, t_features_toplevel_dec),
        axis=2,
        name='input_listener')
    t_pred = tf.squeeze(
        net.mlp(t_out_feats, (n_hidden, 1), (tf.nn.relu, None)),
        name='prediction')
    if asym:
        t_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=t_labels_l, logits=t_pred),
            name='loss')
        return (t_features_raw, t_labels, t_features_raw_l, t_labels_l,
                (t_msg_discrete if discrete else t_msg), t_pred, t_loss)
    else:
        t_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=t_labels, logits=t_pred))
        return (t_features_raw, t_labels, (t_msg_discrete if discrete else
                                           t_msg), t_pred, t_loss)


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

    parser.add_argument(
        '--model',
        type=str,
        choices=['feature', 'end2end'],
        default='end2end',
        help='Model type')

    parser.add_argument(
        '--debug', action='store_true', help='Use tensorflow debugger')

    data_opts = parser.add_argument_group('data', 'options for data gen')
    data_opts.add_argument(
        '--data',
        type=str,
        help='Folder of dataset to load (cannot be used with --components)')
    data_opts.add_argument(
        '--components',
        action='store_true',
        help='Generate dataset from components instead (cannot be used with '
        '--data')

    component_args = parser.add_argument_group(
        'components',
        'options for generating data from cmponents (only supports asym now)')
    component_args.add_argument(
        '--train_components',
        nargs='+',
        default=CONFIGS['shape_color_generalization_3']['train'])
    component_args.add_argument(
        '--test_components',
        nargs='+',
        default=CONFIGS['shape_color_generalization_3']['test'])
    component_args.add_argument(
        '--n_dev', type=int, default=1024, help='Dev set size')
    component_args.add_argument(
        '--n_test',
        type=int,
        default=1024,
        help='Number of testing examples to create from components')

    component_args.add_argument(
        '--asym_max_images',
        default=5,
        type=int,
        help='Maximum images in each asymmetric world')
    component_args.add_argument(
        '--asym_min_targets',
        default=2,
        type=int,
        help='Minimum targets in each asymmetric world')
    component_args.add_argument(
        '--asym_min_distractors',
        default=1,
        type=int,
        help='Minimum distractors in each asymmetric world')

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
    net_opts.add_argument(
        '--rnn_cell',
        type=str,
        default='gru',
        choices=['lstm', 'gru'],
        help='RNN Cell type')
    net_opts.add_argument(
        '--tensorboard',
        action='store_true',
        help='Save tensorboard graph, don\'t do anything else')
    net_opts.add_argument(
        '--tensorboard_messages',
        action='store_true',
        help='Save test (or train if not --test) messages for '
        'tensorboard embedding visualization')
    net_opts.add_argument(
        '--tensorboard_save',
        default='./saves/tensorboard/rnn-syn-graph',
        help='Tensorboard graph save dir')

    train_opts = parser.add_argument_group('train', 'options for net training')
    train_opts.add_argument(
        '--restore', action='store_true', help='Restore model')
    train_opts.add_argument(
        '--restore_path',
        type=str,
        default='saves/{data}-{model}-model.model',
        help='Restore filepath (can use parser options)')
    train_opts.add_argument(
        '--save', action='store_true', help='Save model file')
    train_opts.add_argument(
        '--save_path',
        type=str,
        default='saves/{data}-{model}-model.model',
        help='Save model filepath (can use parser options)')
    train_opts.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (if none, picked randomly)')
    train_opts.add_argument(
        '--tf_seed',
        type=int,
        default=None,
        help='Random TensorFlow seed (by default, same as args.seed)')
    train_opts.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    train_opts.add_argument(
        '--epochs', type=int, default=16, help='Number of training epochs')
    train_opts.add_argument(
        '--max_data',
        type=int,
        default=None,
        help='Max size of training data (rest discarded)')
    train_opts.add_argument(
        '--dev_every',
        type=int,
        default=10,
        help='How often (in epochs) to report dev results. '
             'Only applies to components (for now)')

    test_opts = parser.add_argument_group('train', 'options for net testing')
    test_opts.add_argument('--test', action='store_true', help='do testing')
    test_opts.add_argument(
        '--test_split',
        type=float,
        default=0.2,
        help='%% of dataset to test on')
    test_opts.add_argument(
        '--test_no_unique',
        action='store_true',
        help='Don\'t require testing unique configs')

    save_opts = parser.add_argument_group('save messages')
    save_opts.add_argument(
        '--no_save_msgs',
        action='store_true',
        help='Don\'t save comptued messages after testing')
    save_opts.add_argument(
        '--msgs_file',
        default='data/{data}-{model}-{comm_type}'
        '{n_comm}-{epochs}epochs-msgs.pkl',
        help='Save location (can use parser options)')
    save_opts.add_argument(
        '--save_max',
        type=int,
        default=None,
        help='Maximum number of messages to save')

    args = parser.parse_args()

    if args.components and args.data:
        parser.error("Can't specify --data and --components")
    if not args.components and not args.data:
        parser.error("Must specify one of --data or --components")

    if args.seed is not None:
        random = np.random.RandomState(args.seed)
    else:
        random = np.random.RandomState(args.seed)

    if args.tf_seed is not None:
        tf.set_random_seed(args.tf_seed)
    elif args.seed is not None:
        tf.set_random_seed(args.seed)

    if args.components:
        if any(x in args.train_components for x in args.test_components):
            print("Warning: test components in train components, could be"
                  "repeats depending on size of data")
        # Hardcoded asym
        asym = True
        if asym:
            print("Generating from components (asym)")
        else:
            raise NotImplementedError
        print("Loading training components")
        configs, components_dict = load_components(args.train_components)
        # Generate metadata ourself
        asym_args = {
            'max_images': args.asym_max_images,
            'min_targets': args.asym_min_targets,
            'min_distractors': args.asym_min_distractors
        }
        train = make_from_components(
            args.batch_size,
            configs,
            components_dict,
            asym=asym,
            asym_args=asym_args)
        # To satisfy later args
        metadata = {
            'asym': True,
            'asym_args': asym_args,
        }
        # Generate a dev set
        dev, dev_metadata = zip(*make_from_components(
            args.n_dev,
            configs,
            components_dict,
            asym=asym,
            asym_args=asym_args))
    else:
        print("Loading data")
        train, metadata = swdata.load_scenes(args.data, gz=True)

        asym = False
        if 'asym' in metadata and metadata['asym']:
            print("Asymmetric dataset detected")
            asym = True

        # Train/test split
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
                        unique_sets.append((config_data, config_md))
                random.shuffle(unique_sets)
                train, test = swdata.train_test_split(
                    unique_sets, test_split=args.test_split)
                train = swdata.flatten(train, with_metadata=True)
                test = swdata.flatten(test, with_metadata=True)
                random.shuffle(train)
                random.shuffle(test)
            else:
                train = list(zip(train, metadata['configs']))
                train, test = swdata.train_test_split(
                    train, test_split=args.test_split)
                train = swdata.flatten(train, with_metadata=True)
                test = swdata.flatten(test, with_metadata=True)
                random.shuffle(train)
                random.shuffle(test)
            print("Train:", len(train), "Test:", len(test))
        else:
            # Just train on everything.
            train = list(zip(train, metadata['configs']))
            train = swdata.flatten(train, with_metadata=True)
            random.shuffle(train)

    if asym:
        max_images = metadata['asym_args']['max_images']
        n_attrs = len(train[0].world.speaker_worlds[0].shapes[0])
    else:
        max_images = metadata['n_targets'] + metadata['n_distractors']
        n_attrs = len(train[0].world.worlds[0].shapes[0])

    # Hardcoded for now
    max_shapes = 2

    print("Building model")
    if args.model == 'feature':
        if asym:
            tfs, tls, tfl, tll, t_msg, t_pred, t_loss = build_feature_model(
                max_images,
                max_shapes,
                n_attrs,
                net_arch=(args.n_hidden, args.n_comm),
                discrete=args.comm_type == 'discrete',
                asym=True)
        else:
            t_features, t_labels, t_msg, t_pred, t_loss = build_feature_model(
                max_images,
                max_shapes,
                n_attrs,
                net_arch=(args.n_hidden, args.n_comm),
                discrete=args.comm_type == 'discrete',
                asym=False)
    elif args.model == 'end2end':
        if asym:
            tfs, tls, tfl, tll, t_msg, t_pred, t_loss = build_end2end_model(
                max_images,
                net_arch=(args.n_hidden, args.n_comm, 1024),
                discrete=args.comm_type == 'discrete',
                rnncell=RNN_CELLS[args.rnn_cell],
                asym=True)
        else:
            t_features, t_labels, t_msg, t_pred, t_loss = build_end2end_model(
                max_images,
                net_arch=(args.n_hidden, args.n_comm, 1024),
                discrete=args.comm_type == 'discrete',
                rnncell=RNN_CELLS[args.rnn_cell],
                asym=False)
    else:
        raise RuntimeError("Unknown model type {}".format(args.model))
    optimizer = tf.train.AdamOptimizer(0.001)
    o_train = optimizer.minimize(t_loss)
    session = tf.Session()
    if args.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(
            session, dump_root='/local/scratch/jlm95/tfdbg/')
    session.run(tf.global_variables_initializer())

    if args.tensorboard:
        print("Saving logs to {}".format(args.tensorboard_save))
        tf.summary.FileWriter(
            args.tensorboard_save, graph=tf.get_default_graph())
        print("Exiting")
        sys.exit(0)

    # ==== TRAIN ====
    if args.restore:
        saver = tf.train.Saver()
        saver.restore(session, args.restore_path.format(**vars(args)))
    else:
        acc_history = []
        loss_history = []

        print("Training")
        for epoch in range(args.epochs):
            if args.components:
                if epoch != 0:
                    # Sample new components
                    train = make_from_components(
                        args.batch_size,
                        configs,
                        components_dict,
                        asym=asym,
                        asym_args=asym_args)
            else:
                # Shuffle training data, since epoch is complete
                random.shuffle(train)
            loss = 0
            hits = 0
            total = 0
            if args.components:
                batch_iter = [train]
            else:
                batch_iter = batches(
                    train, args.batch_size, max_data=args.max_data)
            for batch in batch_iter:
                batch, batch_metadata = zip(*batch)
                if args.model == 'feature':
                    if asym:
                        # Since we need to measure accuracy stats on listener
                        # labels, keep name for those
                        se, sl, envs, labels = swdata.extract_envs_and_labels(
                            batch, max_images, max_shapes, n_attrs, asym=True)
                    else:
                        envs, labels = swdata.extract_envs_and_labels(
                            batch, max_images, max_shapes, n_attrs, asym=False)
                elif args.model == 'end2end':
                    if asym:
                        se, sl, envs, labels = swdata.prepare_end2end(
                            batch, max_images, asym=True)
                    else:
                        envs, labels = swdata.prepare_end2end(
                            batch, max_images, asym=False)
                else:
                    raise RuntimeError
                if asym:
                    l, preds, _ = session.run([t_loss, t_pred, o_train], {
                        tfs: se,
                        tls: sl,
                        tfl: envs,
                        tll: labels
                    })
                else:
                    l, preds, _ = session.run([t_loss, t_pred, o_train], {
                        t_features: envs,
                        t_labels: labels
                    })

                match = (preds > 0) == labels
                loss += l
                hits += np.all(match, axis=1).sum()
                total += len(match)

            if args.components and (epoch % args.dev_every == 0):
                # Every 10 epochs, print dev accuracy
                if args.model == 'feature':
                    if asym:
                        # Since we need to measure accuracy stats on listener
                        # labels, keep name for those
                        dev_se, dev_sl, dev_envs, dev_labels = swdata.extract_envs_and_labels(
                            dev, max_images, max_shapes, n_attrs, asym=True)
                    else:
                        dev_envs, dev_labels = swdata.extract_envs_and_labels(
                            dev, max_images, max_shapes, n_attrs, asym=False)
                elif args.model == 'end2end':
                    if asym:
                        dev_se, dev_sl, dev_envs, dev_labels = swdata.prepare_end2end(
                            dev, max_images, asym=True)
                    else:
                        dev_envs, dev_labels = swdata.prepare_end2end(
                            dev, max_images, asym=False)
                else:
                    raise RuntimeError
                if asym:
                    dev_l, dev_preds, dev_msgs = session.run(
                        [t_loss, t_pred, t_msg], {
                        tfs: dev_se,
                        tls: dev_sl,
                        tfl: dev_envs,
                        tll: dev_labels
                    })
                else:
                    dev_l, dev_preds, dev_msgs = session.run(
                        [t_loss, t_pred, t_msg], {
                        t_features: dev_envs,
                        t_labels: dev_labels
                    })

                dev_match = (dev_preds > 0) == dev_labels
                dev_hits = np.all(match, axis=1).sum()
                dev_acc = dev_hits / args.n_dev
                print("Epoch {}: Dev accuracy {}, Loss {}".format(
                    epoch, dev_acc, dev_l))
            elif not args.components:
                acc = hits / total
                print("Epoch {}: Accuracy {}, Loss {}".format(
                    epoch, acc, loss))

                loss_history.append(loss)
                acc_history.append(acc)

        if args.save:
            saver = tf.train.Saver()
            saver.save(session, args.save_path.format(**vars(args)))

    # ==== TEST ====
    if args.components:
        if not args.test:
            print("Warning: --components but not --test, using dev")
            test_or_train = zip(dev, dev_metadata)
        else:
            print("Loading testing components")
            # Make sure memory is free
            del configs, components_dict
            gc.collect()
            configs, components_dict = load_components(args.test_components)
            test_or_train = make_from_components(
                args.n_test,
                configs,
                components_dict,
                asym=asym,
                asym_args=asym_args)
    else:
        test_or_train = test if args.test else train

    # Eval test in batches too
    print("Eval test")
    all_records = []

    if args.components:
        # Add dev messages
        dev_true_examples = list(map(find_true_example, zip(dev_se, dev_sl)))

        dev_records = zip(
            dev_msgs,
            dev_preds,
            dev_labels,
            (x.relation[0] for x in dev),
            (x.relation_dir for x in dev),
            (c['target'][0] for c in dev_metadata),
            (c['target'][1] for c in dev_metadata),
            (c['distractor'][0] for c in dev_metadata),
            (c['distractor'][1] for c in dev_metadata),
            dev_true_examples,
            'dev')
        all_records.extend(dev_records)



    for batch in batches(test_or_train, args.batch_size):
        batch, batch_metadata = zip(*batch)
        if args.model == 'feature':
            if asym:
                bse, bsl, batch_envs, batch_labels = \
                    swdata.extract_envs_and_labels(
                        batch, max_images, max_shapes, n_attrs, asym=True)
            else:
                batch_envs, batch_labels = swdata.extract_envs_and_labels(
                    batch, max_images, max_shapes, n_attrs, asym=False)
        elif args.model == 'end2end':
            if asym:
                bse, bsl, batch_envs, batch_labels = \
                    swdata.prepare_end2end(batch, max_images, asym=True)
            else:
                batch_envs, batch_labels = swdata.prepare_end2end(
                    batch, max_images, asym=False)
        else:
            raise RuntimeError("Unknown model {}".format(args.model))

        if asym:
            batch_msgs, batch_preds = session.run([t_msg, t_pred], {
                tfs: bse,
                tls: bsl,
                tfl: batch_envs,
                tll: batch_labels
            })
        else:
            batch_msgs, batch_preds = session.run([t_msg, t_pred], {
                t_features: batch_envs,
                t_labels: batch_labels
            })

        bse_true_examples = list(map(find_true_example, zip(bse, bsl)))

        batch_records = zip(
            batch_msgs,
            batch_preds,
            batch_labels,
            (x.relation[0] for x in batch),
            (x.relation_dir for x in batch),
            (c['target'][0] for c in batch_metadata),
            (c['target'][1] for c in batch_metadata),
            (c['distractor'][0] for c in batch_metadata),
            (c['distractor'][1] for c in batch_metadata),
            # BSE
            bse_true_examples,
            'test')
        all_records.extend(batch_records)

    all_df = pd.DataFrame.from_records(
        all_records,
        columns=('msg', 'pred', 'obs', 'relation', 'relation_dir',
                 'target_shape', 'target_color', 'distractor_shape',
                 'distractor_color', 'example_image', 'phase'))
    all_df.pred = all_df.pred.apply(lambda x: x > 0)
    all_df.obs = all_df.obs.apply(lambda x: x.astype(np.bool))
    all_df['correct'] = pd.Series(
        map(lambda t: np.all(t[0] == t[1]), zip(all_df.pred, all_df.obs)),
        dtype=np.bool)
    all_df.relation = all_df.relation.astype('category')
    all_df.relation_dir = all_df.relation_dir > 0
    for cat_col in [
            'target_shape', 'target_color', 'distractor_shape',
            'distractor_color'
    ]:
        all_df[cat_col] = all_df[cat_col].astype('category')

    if args.test:  # Print test accuracy
        print("Test accuracy: {}".format(all_df.correct.mean()))

    if args.save_max is not None:
        all_df = all_df.iloc[:args.save_max]

    if not args.no_save_msgs:
        print("Saving {} model predictions".format(all_df.shape[0]))
        all_df.to_pickle((args.msgs_file.format(**vars(args))))

    if args.tensorboard_messages:
        # Number of messages limited by sprite size
        ind_size = 64
        sprite_size = 8192  # Use slightly less for perf
        os.makedirs(args.tensorboard_save, exist_ok=True)
        if all_df.shape[0] > ((sprite_size / ind_size)**2):
            print(
                "Warning: too many images, will truncate. Increase sprite size!"
            )
            all_df = all_df.iloc[:(sprite_size / ind_size)**2]

        from tensorflow.contrib.tensorboard.plugins import projector
        msg_combined = np.vstack(all_df.msg)
        messages = tf.Variable(
            tf.convert_to_tensor(
                msg_combined,
                name='messages_embed_raw',
                preferred_dtype=np.float32),
            name='messages_embed')

        # Save messages to model checkpoint
        saver = tf.train.Saver([messages])
        session.run(messages.initializer)
        saver.save(session, os.path.join(args.tensorboard_save, "model.ckpt"))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = messages.name

        # Save metadata
        md_path = os.path.join(args.tensorboard_save, 'metadata.tsv')
        md_df = all_df[[
            'correct', 'target_color', 'target_shape', 'distractor_color',
            'distractor_shape', 'relation', 'relation_dir', 'phase'
        ]]

        # Make target/distractor strings too
        md_df['target'] = pd.Series(
            ['{}-{}'.format(x, y)
             for x, y in zip(md_df.target_shape, md_df.target_color)])
        md_df['distractor'] = pd.Series(
            ['{}-{}'.format(x, y)
             for x, y in zip(md_df.distractor_shape,
                             md_df.distractor_color)])
        md_df['config'] = pd.Series(
            ['{}-{}'.format(x, y)
             for x, y in zip(md_df.target,
                             md_df.distractor)])
        md_df.to_csv(md_path, sep='\t', index=False)

        embedding.metadata_path = 'metadata.tsv'

        # Sprites
        sprite_path = os.path.join(args.tensorboard_save, 'sprite.png')
        # Assume 64px sprites
        embedding.sprite.image_path = 'sprite.png'
        embedding.sprite.single_image_dim.extend([ind_size, ind_size])
        sprite_arr = np.zeros((sprite_size, sprite_size, 3), dtype=np.float32)
        ex_img_i = 0
        try:
            for si in range(0, sprite_size, ind_size):
                for sj in range(0, sprite_size, ind_size):
                    ex_img = all_df.example_image[ex_img_i]
                    sprite_arr[si:si + ind_size, sj:sj + ind_size] = ex_img
                    ex_img_i += 1
        except KeyError:
            assert ex_img_i == len(all_df.example_image)
        sprite_arr = (sprite_arr * 255).astype(np.uint8)
        imsave(sprite_path, sprite_arr)

        summary_writer = tf.summary.FileWriter(args.tensorboard_save)
        projector.visualize_embeddings(summary_writer, config)
