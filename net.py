import tensorflow as tf
import contextlib

INIT_SCALE = 1.47


def linear(t_in, n_out, name_scope=None):
    if name_scope is None:
        ctx = contextlib.ExitStack()
    else:
        ctx = tf.name_scope(name_scope)
    with ctx:
        if len(t_in.get_shape()) == 2:
            op = "ij,jk->ik"
        elif len(t_in.get_shape()) == 3:
            op = "ijk,kl->ijl"
        else:
            assert False
        v_w = tf.get_variable(
            "w",
            shape=(t_in.get_shape()[-1], n_out),
            initializer=tf.initializers.variance_scaling(
                scale=INIT_SCALE, distribution='uniform'))
        v_b = tf.get_variable(
            "b", shape=n_out, initializer=tf.constant_initializer(0))
        return tf.einsum(op, t_in, v_w) + v_b


def mlp(t_in, widths, activations, name_scope=None):
    if name_scope is None:
        ctx = contextlib.ExitStack()
    else:
        ctx = tf.name_scope(name_scope)
    assert len(widths) == len(activations)
    with ctx:
        prev_layer = t_in
        for i_layer, (width, act) in enumerate(zip(widths, activations)):
            with tf.variable_scope('mlp_{}'.format(str(i_layer))):
                layer = linear(prev_layer, width)
                if act is not None:
                    layer = act(layer)
            prev_layer = layer
        return prev_layer


def convolve(raw, n_images, n_toplevel_conv, name_scope=None):
    if name_scope is None:
        ctx = contextlib.ExitStack()
    else:
        ctx = tf.name_scope(name_scope)
    with ctx:
        conv1_list = []
        for i in range(n_images):
            slice = raw[:, i, :, :, :]
            t_conv_1 = tf.layers.conv2d(slice, filters=32,
                                        strides=[2, 2],
                                        kernel_size=[5, 5],
                                        padding='same', activation=tf.nn.relu)
            conv1_list.append(t_conv_1)

        conv1 = tf.stack(conv1_list, axis=1)

        # Second convolutional layer - 64 features with a 5x5 filter, stride 2
        conv2_list = []
        for i in range(n_images):
            slice = conv1[:, i, :, :, :]
            t_conv_2 = tf.layers.conv2d(slice, filters=64,
                                        strides=[2, 2],
                                        kernel_size=[5, 5],
                                        padding='same', activation=tf.nn.relu)
            conv2_list.append(t_conv_2)

        conv2 = tf.stack(conv2_list, axis=1)

        conv2_flat = tf.reshape(
            conv2,
            (-1, n_images, conv2.shape[2] * conv2.shape[3] * conv2.shape[4]))

        # Dense layer
        dense = tf.layers.dense(inputs=conv2_flat, units=n_toplevel_conv,
                                activation=tf.nn.relu)
        return dense
