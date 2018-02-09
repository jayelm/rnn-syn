"""
Run rnn-syn experiments.
"""

import net
import tensorflow as tf
import numpy as np
import swdata
from swdata import AsymScene, Scene, SWorld, TrainEx
import sys
import time
from tensorflow.python import debug as tf_debug
import pandas as pd

RNN_CELLS = {
    'gru': tf.contrib.rnn.GRUCell,
    'lstm': tf.contrib.rnn.LSTMCell,
}

assert AsymScene
assert Scene
assert SWorld
assert TrainEx


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
        '--data', type=str, required=True, help='Folder of dataset to load')

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
        '--tensorboard_save',
        default='./rnn-syn-graph',
        help='Tensorboard graph save file')

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
        default='{data}-{model}-{rnn_cell}-{comm_type}{n_comm}-'
        '{epochs}epochs-msgs.pkl',
        help='Save location (can use parser options)')
    save_opts.add_argument(
        '--save_max',
        type=int,
        default=None,
        help='Maximum number of messages to save')

    args = parser.parse_args()

    if args.seed is not None:
        random = np.random.RandomState(args.seed)
    else:
        random = np.random.RandomState(args.seed)

    if args.tf_seed is not None:
        tf.set_random_seed(args.tf_seed)
    elif args.seed is not None:
        tf.set_random_seed(args.seed)

    print("Loading data")
    train, metadata = swdata.load_scenes(args.data, gz=True)

    asym = False
    if 'asym' in metadata and metadata['asym']:
        print("Asymmetric dataset detected")
        asym = True

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
                    unique_sets.append((config_data, config_md))
            random.shuffle(unique_sets)
            train, test = swdata.train_test_split(
                unique_sets, test_split=args.test_split)
            train = swdata.flatten(train, with_metadata=True)
            test = swdata.flatten(test, with_metadata=True)
            random.shuffle(train)
            random.shuffle(test)
        else:
            train = list(map(TrainEx, zip(train, metadata['configs'])))
            train, test = swdata.train_test_split(
                train, test_split=args.test_split)
            train = swdata.flatten(train, with_metadata=True)
            test = swdata.flatten(test, with_metadata=True)
            random.shuffle(train)
            random.shuffle(test)
        print("Train:", len(train), "Test:", len(test))
    else:
        # Just train on everything
        train = list(map(TrainEx, zip(train, metadata['configs'])))
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
            loss = 0
            hits = 0
            total = 0
            # Shuffle training data
            random.shuffle(train)
            for batch in batches(
                    train, args.batch_size, max_data=args.max_data):
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

            acc = hits / total
            print("Epoch {}: Accuracy {}, Loss {}".format(epoch, acc, loss))

            loss_history.append(loss)
            acc_history.append(acc)

        if args.save:
            saver = tf.train.Saver()
            saver.save(session, args.save_path.format(**vars(args)))

    # ==== TEST ====
    test_or_train = test if args.test else train

    # Eval test in batches too
    all_records = []

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
        )
        batch_records = list(batch_records)  # TEMP
        all_records.extend(batch_records)

    all_df = pd.DataFrame.from_records(
        all_records,
        columns=('msg', 'pred', 'obs', 'relation', 'relation_dir',
                 'target_shape', 'target_color', 'distractor_shape',
                 'distractor_color'))
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

    # broadcast the input message over scenes
    #  tile_msg = np.asarray([msg for _ in range(n_sample)])
    # sample environments from the given scene ids
    # In this case - we need our own "theories". Or just sample a bunch, and
    # greenlight/badlight em so we can see what the messages mean
    # In particular, we may need to do clustering on the messages.
    # Actually, we may want to cluster the messages in ipynb, once we've analyzed
    # Then we can add a --eval_msgs argument to this script
    #  envs, _ = sample_annotated(dataset, random.choice(scene_ids, size=n_sample).tolist())
    # This is used to eval messages given labels - find @ rnn-syn.ipynb
    #  model_preds, = session.run([t_pred], {t_features: envs, t_msg: tile_msg})

    if not args.no_save_msgs:
        print("Saving {} model predictions".format(all_df.shape[0]))
        all_df.to_pickle((args.msgs_file.format(**vars(args))))
