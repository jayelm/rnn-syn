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


# Cell Factory
def create_conv_cell(inputs, kernel_size, stride, cell_name):
    """Create a single convolutional operator cell"""
    assert len(kernel_size) == 4, "Kernel size %s is incorrectly specified" % kernel_size
    assert len(stride) == 4, "Stride %s is incorrectly specified" % stride
    
    kernel = tf.get_variable(name=cell_name + "_kernel", shape=kernel_size)
    bias = tf.get_variable(name=cell_name + "_bias", shape=(kernel_size[-1]))
    
    conv = tf.nn.conv2d(inputs, kernel, stride, padding="SAME", name=cell_name + "_conv")
    conv = tf.nn.bias_add(conv, bias, name=cell_name + "_bias")
    conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=True)
    conv = tf.nn.relu(conv, name=cell_name + "_relu")
    
    return conv

# 3-deep layer factory
def create_conv_layer(inputs, filter_size, prev_channels, num_filters, stride_len, pool, layer_name):
    """Create a set of 3 conv_cells all with the same stride and filter size"""
    initial_kernel = [filter_size, filter_size, prev_channels, num_filters]
    later_kernel = [filter_size, filter_size, num_filters, num_filters]
    
    stride = [1, stride_len, stride_len, 1]
    cell1 = create_conv_cell(inputs, initial_kernel, stride, layer_name + "_c1")
    cell2 = create_conv_cell(cell1, later_kernel, stride, layer_name + "_c2")
    cell3 = create_conv_cell(cell2, later_kernel, stride, layer_name + "_c3")
    
    if pool:
        cell3 = tf.nn.max_pool(cell3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                               padding="SAME", name=layer_name + "_maxpool")
    return cell3

# Number of channels for filter input and output
nfilt = (3, 32, 64)




def convolve(raw, n_images, n_toplevel_conv, var_scope=None):
    assert var_scope is not None
    with tf.variable_scope(var_scope, initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
        gmps = []
        for i in range(n_images):
            slice = raw[:, i, :, :, :]
            layer1 = create_conv_layer(slice, filter_size=5, prev_channels=nfilt[0], num_filters=nfilt[1],
                                       stride_len=1, pool=True, layer_name="conv1")
            
            layer2 = create_conv_layer(layer1, filter_size=3, prev_channels=nfilt[1], num_filters=nfilt[2],
                                       stride_len=1, pool=True, layer_name="conv2")
            
            #  layer3 = create_conv_layer(layer2, filter_size=3, prev_channels=nfilt[2], num_filters=nfilt[3],
                                       #  stride_len=1, pool=False, layer_name="conv3")
            
            # Global max pooling over the final conv cells
            gmp0 = tf.reduce_max(layer2, [1, 2], name="cnn_end")
            gmps.append(gmp0)

        gmp_global = tf.stack(gmps, axis=1)
        return gmp_global
        #  print(gmp_global.shape)

        #  raise Exception

        #  # Dense layer
        #  dense = tf.layers.dense(inputs=gmp_flat, units=n_toplevel_conv,
                                #  activation=tf.nn.relu)
        #  return dense
