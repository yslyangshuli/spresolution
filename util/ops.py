'''
通用的卷积操作
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim


def GetKernel(name, shape, trainable=True):
    return tf.get_variable(name, shape, trainable=trainable)

'''
指定卷积核卷积
'''
def Conv2d(input, kernel, normalization=slim.batch_norm,
           strides=1, padding='SAME', activation_fn=None, scope='conv2d'):
    with tf.variable_scope(scope):
        net = tf.nn.conv2d(input, kernel, strides=[1, strides, strides, 1], padding=padding, name='con2d')
        if normalization:
            net = normalization(net, scope='bn')
        if activation_fn:
            net = activation_fn(net)
        return net

'''
上采样
'''
def PixelShuffler(input, kernel, ratios=2, scope='pixshuffler'):
    with tf.variable_scope(scope):
        net = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME', name='upsample')
        net = tf.depth_to_space(net, block_size=ratios)
        return net

def Deconv2d(input, kernel, stride=2, padding='SAME', scope='deconv2d'):
    #input=[batch, h, w, channel], kernel = [kernel_size, kernel_size, indim, outdim]
    #input.channel = kernel.indim
    outdim = kernel.get_shape()[-1]
    [b, h, w, c] = input.get_shape()
    with tf.variable_scope(scope):
        net = tf.nn.conv2d_transpose(input, kernel, [b, h * 2, w * 2, outdim],
                                     strides=[1, stride, stride, 1], padding=padding)
        return net
