"""
RNN SYN 2: streamlined version
"""

import net
import tensorflow as tf
import shapeworld_fast as sf
import numpy as np
import random
import multiprocessing as mp
import time
import os


def shuffle_envs_labels(envs, labels):
    new_envs = np.zeros_like(envs)
    new_labels = np.zeros_like(labels)
    world_seq = list(range(envs[0].shape[0]))
    # Loop through each world (env/label) in the batch
    for env_i, (env, label) in enumerate(zip(envs, labels)):
        # New sequence of worlds to retrieve from original envs/labels
        random.shuffle(world_seq)
        # Loop through new sequence, place this sequence in increasing order
        # in new_envs/labels
        for new_world_i, orig_world_i in enumerate(world_seq):
            new_envs[env_i, new_world_i] = env[orig_world_i]
            new_labels[env_i, new_world_i] = label[orig_world_i]
    return new_envs, new_labels


def build_end2end_model(n_images,
                        image_dim=(64, 64, 3),
                        net_arch=(256, 64, 1024),
                        rnncell=tf.contrib.rnn.GRUCell):
    """
    Return an encoder-decoder model that uses raw ShapeWorld images.

    Listener has separate input and labels, since listener must be shuffled.

    net_arch:
        (number of GRU hidden units, message dimensionality,
         convnet toplevel layer dimensionality)

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

    # Decoder makes independent predictions for each set of object features
    with tf.name_scope('message_process'):
        t_expand_msg = tf.expand_dims(t_msg, axis=1)
        t_tile_message = tf.tile(t_expand_msg, (1, n_images, 1))

    t_features_toplevel_dec = net.convolve(t_features_raw_l, n_images,
                                           n_toplevel_conv, 'conv_listener')
    t_out_feats = tf.concat(
        (t_tile_message, t_features_toplevel_dec),
        axis=2,
        name='input_listener')
    t_pred = tf.squeeze(
        net.mlp(t_out_feats, (n_hidden, 1), (tf.nn.relu, None)),
        name='prediction')
    t_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=t_labels_l, logits=t_pred),
        name='loss')
    loss_summary = tf.summary.scalar('loss', t_loss)
    return (t_features_raw, t_labels, t_features_raw_l, t_labels_l, t_msg,
            t_pred, t_loss, t_features_toplevel_enc, t_features_toplevel_dec,
            loss_summary)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='RNN SYN 2', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--max_images', type=int, default=20, help='Maximum number of images')
    parser.add_argument('--n_batches', type=int, default=1000)
    now = time.strftime('%Y-%m-%d-%X', time.localtime())
    parser.add_argument('--log_dir', default=os.path.join('./logs', now),
                        help='Tensorboard log directory')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_cpu', type=int, default=mp.cpu_count())
    parser.add_argument(
        '--correct_proportion',
        type=float,
        default=0.5,
        help='Correct proportion')

    args = parser.parse_args()

    t_feat_spk, t_lab_spk, t_feat_lis, t_lab_lis, t_msg, t_pred, t_loss, t_conv_s, t_conv_l, loss_summary_op = build_end2end_model(
        args.max_images)

    optimizer = tf.train.AdamOptimizer()
    o_train = optimizer.minimize(t_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph())

    # Init pool here
    pool = mp.Pool(args.n_cpu)

    for batch_i in range(args.n_batches):
        feat_spk, lab_spk, configs = sf.generate(
            args.batch_size,
            args.max_images,
            args.correct_proportion,
            float_type=True,
            pool=pool)

        # Shuffle images for listener
        feat_lis, lab_lis = shuffle_envs_labels(feat_spk, lab_spk)

        batch_loss, preds, _, loss_summary = session.run(
            [t_loss, t_pred, o_train, loss_summary_op], {
                t_feat_spk: feat_spk,
                t_lab_spk: lab_spk,
                t_feat_lis: feat_lis,
                t_lab_lis: lab_lis
            })

        match = (preds > 0) == lab_lis
        match_acc = np.mean(match)
        hits = np.all(match, axis=1).mean()
        if batch_i % 10 == 0:
            print("Batch {}: overall acc: {:.4f} hits only: {:.4f} loss: {:.4f}".format(
                batch_i, match_acc, hits, batch_loss))

        # Add summaries
        partial_acc_summary = tf.Summary(value=[
            tf.Summary.Value(tag='partial_acc',
                             simple_value=match_acc)
        ])
        hits_summary = tf.Summary(value=[
            tf.Summary.Value(tag='hits',
                             simple_value=hits)
        ])
        writer.add_summary(partial_acc_summary, batch_i)
        writer.add_summary(hits_summary, batch_i)
        writer.add_summary(loss_summary, batch_i)

    pool.close()
    pool.join()
