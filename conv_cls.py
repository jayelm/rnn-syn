from rnn_syn import CONFIGS
from swdata import load_components, invert
import tensorflow as tf
import net
from random import choice
import numpy as np


FEATS = np.array([
    'red',
    'green',
    'blue',
    'yellow',
    'magenta',
    'cyan',
    'square',
    'cross',
    'circle',
    'triangle'
])
FEATS_MAP = invert(dict(enumerate(FEATS)))


def make_batch(n, configs, components_dict):
    imgs = []
    labs = []
    for _ in range(n):
        this_lab = np.zeros_like(FEATS, dtype=np.float32)
        config = choice(configs)
        target, distractor, rel = config
        t_or_d = choice([components_dict[config]['targets'],
                         components_dict[config]['distractors']])
        ans = choice(t_or_d)[0]
        for td in [target, distractor]:
            for i in range(2):
                this_lab[FEATS_MAP[td[i]]] = 1.0
        imgs.append(ans)
        labs.append(this_lab)

    return np.stack(imgs), np.array(labs)


if __name__ == '__main__':
    # Convnet architecture
    t_features_raw = tf.placeholder(
        tf.float32, (None, 64, 64, 3), name='features_speaker')

    # Conv layer 1
    t_conv_1 = tf.layers.conv2d(t_features_raw, filters=32,
                                strides=[2, 2],
                                kernel_size=[5, 5],
                                padding='same', activation=tf.nn.relu)

    # Conv layer 2
    t_conv_2 = tf.layers.conv2d(t_conv_1, filters=64, strides=[2, 2],
                                kernel_size=[5, 5], padding='same', activation=tf.nn.relu)

    # Flatten conv layer 2
    t_conv_2_flat = tf.reshape(t_conv_2, (-1, t_conv_2.shape[1] * t_conv_2.shape[2] * t_conv_2.shape[3]))

    # Dense (for enc)
    t_features_toplevel_enc = tf.layers.dense(inputs=t_conv_2_flat, units=1024, activation=tf.nn.relu)

    # Actual prediction (not in original model) - can toplevel enc predict
    # features? (multiclass problem)
    t_pred = tf.layers.dense(inputs=t_features_toplevel_enc, units=len(FEATS))


    t_labels = tf.placeholder(tf.float32, (None, len(FEATS)))

    t_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=t_labels, logits=t_pred))

    optimizer = tf.train.AdamOptimizer()
    o_train = optimizer.minimize(t_loss)

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    # Load configs
    comps = CONFIGS['shape_color_generalization_5']['train'] + CONFIGS['shape_color_generalization_5']['test']
    configs, components_dict = load_components(comps, maxdata=100)

    N_DEV = 512
    dev_batch, dev_labs = make_batch(N_DEV, configs, components_dict)

    for n in range(10000):
        batch, labs = make_batch(128, configs, components_dict)
        loss, preds, _ = session.run([t_loss, t_pred, o_train], {
            t_features_raw: batch,
            t_labels: labs
        })

        if (n % 10) == 0:
            dev_loss, dev_preds = session.run([t_loss, t_pred], {
                t_features_raw: dev_batch,
                t_labels: dev_labs
            })
            dev_match = (dev_preds > 0) == dev_labs
            dev_hits = np.all(dev_match, axis=1).sum()
            dev_acc = dev_hits / N_DEV
            print("Epoch {}: Dev accuracy {}, Loss {}".format(
                n, dev_acc, dev_loss))
