# Code taken from https://github.com/openai/baselines/tree/master/baselines/a2c

import os
import numpy as np
import tensorflow as tf

def mse(pred, target):
    return tf.square(pred-target)/2.

def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def conv_without_bias(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, reuse=False, trainable=True):
    with tf.variable_scope(scope, reuse=reuse):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [rf, rf, nin, nf], initializer=ortho_init(init_scale), trainable=trainable)
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad), w

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, reuse=False, trainable=True):
    with tf.variable_scope(scope, reuse=reuse):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [rf, rf, nin, nf], initializer=ortho_init(init_scale), trainable=trainable)
        b = tf.get_variable("b", [nf], initializer=tf.constant_initializer(0.0), trainable=trainable)
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)+b

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0, reuse=False, trainable=True):
    with tf.variable_scope(scope, reuse=reuse):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale), trainable=trainable)
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias), trainable=trainable)
        return tf.matmul(x, w)+b

def constrained_fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0, reuse=False, trainable=True):
    with tf.variable_scope(scope, reuse=reuse):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale), trainable=trainable)
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias), trainable=trainable)

        constrained_w = tf.nn.l2_normalize(w, dim=1) * init_scale
        print('constrained_w: {}'.format(constrained_w.get_shape()))

        return tf.matmul(x, constrained_w)+b

def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

schedules = {
    'linear': lambda p: 1 - p
}

class Scheduler(object):

    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v*self.schedule(self.n/self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v*self.schedule(steps/self.nvalues)
