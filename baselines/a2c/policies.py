import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, conv_without_bias, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, mse, cat_entropy
from baselines.common.distributions import make_pdtype

def nature_cnn_h3(unscaled_images, first_layer_mode='', trainable=True, conv1_fn=lambda x: x):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu

    print("scaled_images: {}".format(scaled_images))

    if first_layer_mode == 'Share':
        assert False
        # input_activations = []
        # for start in range(3):
        #     input_images = scaled_images[..., start:start+2]
        #     assert input_images.get_shape()[-1] == 2 # Should be a pair of frames.

        #     h = activ(conv(input_images, 'c1_2_frame_input', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), reuse=start!=0, trainable=trainable))
        #     input_activations.append(h)

        # assert len(input_activations) == 3 # Should have 3 pairs of frames.
        # h = (1. / 3.) * tf.add_n(input_activations, name='c1') # Average the activations of the three pairs of frames.

    elif first_layer_mode == '2Frame':
        input_images = scaled_images[..., -2:]
        assert input_images.get_shape()[-1] == 2 # Should be a pair of frames.
        h = activ(conv(input_images, 'c1_2_frame_input', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), trainable=trainable))

    else:
        assert False
        # scaled_images = scaled_images[..., -2:]
        # print("scaled_images: {}".format(scaled_images))

        # h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), trainable=trainable))

        # print('scaled_images: {}'.format(scaled_images.get_shape()))

    h = conv1_fn(h)
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), trainable=trainable))
    _h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), trainable=trainable)
    h3 = activ(_h3)
    return h3, _h3

def nature_cnn(unscaled_images, first_layer_mode='', trainable=True):
    h3, _h3 = nature_cnn_h3(unscaled_images, first_layer_mode, trainable)
    h3 = conv_to_fc(h3)
    return tf.nn.relu(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), trainable=trainable))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):
    def __init__(self, sess, X, ob_space, ac_space, nbatch, nsteps, reuse=False, hparams=None): #pylint: disable=W0613
        assert hparams != None
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        self.X = X

        scope = hparams.get('_policy_scope', 'model')

        with tf.variable_scope(scope, reuse=reuse):
            # if '_teacher_h' in hparams:
            #     h = hparams['_teacher_h']
            # else:
            #     h = nature_cnn(X, first_layer_mode=hparams['first_layer_mode'], trainable=hparams['base_trainable'])

            # for i in range(hparams['fc_depth']):
            #     h = tf.nn.relu(fc(h, 'additional_fc{}'.format(i), nh=512, init_scale=np.sqrt(2)))

            h = nature_cnn(X, first_layer_mode=hparams['first_layer_mode'], trainable=hparams['base_trainable'])
            self.original_h = h

            if hparams['use_extra_path']:
                with tf.variable_scope('model_extrapath', reuse=reuse):
                    notransfer_h = nature_cnn(X, first_layer_mode=hparams['first_layer_mode'], trainable=hparams['base_trainable'])
                    print("notransfer_h: {}".format(notransfer_h))
                    print('original h: {}'.format(h))

                    concatenated_h = tf.concat([h, notransfer_h], axis=-1)
                    print('concatenated_h: {}'.format(concatenated_h))

                    h = tf.nn.relu(fc(concatenated_h, 'extra_path_fc', nh=512, init_scale=np.sqrt(2), trainable=True))
                    print('final h: {}'.format(h))

            self.h = h
            init_scales = [0.01, 1]

            if hparams.get('init_pi_v_zero'):
                init_scales = [0 for _ in init_scales]

            pi = fc(h, 'pi', nact, init_scale=init_scales[0])
            vf = fc(h, 'v', 1, init_scale=init_scales[1])[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        # copy_weights_op = None
        # def setup_copy_weights():
        #     nonlocal copy_weights_op
        #     assert hparams.get('_src_scope')

        #     layer_names = ['c1_2_frame_input', 'c2', 'c3', 'fc1', 'pi', 'v']
        #     layer_types = ['w', 'b']
        #     assert hparams['fc_depth'] == 0

        #     src_scope = hparams['_src_scope']
        #     assign_ops = []

        #     for layer_name in layer_names:
        #         for layer_type in layer_types:
        #             var_name = layer_name + '/' + layer_type + ':0'
        #             dst_var_name = scope + '/' + var_name
        #             src_var_name = src_scope + '/' + var_name

        #             dst_var = None
        #             src_var = None

        #             for v in tf.global_variables():
        #                 if v.name == dst_var_name: dst_var = v
        #                 if v.name == src_var_name: src_var = v

        #             assert dst_var is not None
        #             assert src_var is not None

        #             assign_op = tf.assign(ref=dst_var, value=src_var)
        #             assign_ops.append(assign_op)

        #     copy_weights_op = tf.group(*assign_ops)

        # def copy_weights():
        #     sess.run(copy_weights_op)
        # self.copy_weights = copy_weights
        # self.setup_copy_weights = setup_copy_weights


        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        



class CnnAttentionPolicy(object):

    def _init_attention_20(self, conv1):
        if not self.hparams.get('attention_20'): return conv1

        print("INIT ATTENTION 20!!")

        attention_logits, attention_selector = conv_without_bias(conv1, 'attention_logits_20', nf=1, rf=1, stride=1, trainable=True)
        attention_logits_20 = attention_logits

        orig_attention_shape = attention_logits.get_shape()
        attention_logits = tf.reshape(attention_logits, (self.nbatch, -1))
        attention_weights = tf.nn.softmax(attention_logits / self.hparams['attention_temperature'])
        attention_weights = tf.reshape(attention_weights, orig_attention_shape)
        attention_weights_20 = attention_weights

        # self.context_20 = tf.reduce_sum(tf.reduce_sum(self.attention_weights_20 * conv1, axis=1), axis=1)
        # assert(len(self.context_20.shape) == 2)
        # assert(self.context_20.shape[-1] == 32)

        if self.reuse:
            tf.summary.image('predicted_attention_20', attention_weights_20, max_outputs=1)

        if self.hparams['predict_attention_only']:
            return conv1
        if self.hparams['attention_skip']:
            branch1 = conv1 * attention_weights * 400.0
            branch2 = conv1
            return branch1 * self.hparams['attention_skip_c'] + (1. - self.hparams['attention_skip_c']) * branch2

        return attention_logits_20, attention_weights_20, conv1 * attention_weights * 400.0

    def _init_attention_20_main_branch(self, conv1):
        self.attention_logits_20, self.attention_weights_20, attention_weights = self._init_attention_20(conv1)
        return attention_weights

    def _init_attention_20_extrapath(self, conv1):
        self.extrapath_attention_logits_20, self.extrapath_attention_weights_20, attention_weights = self._init_attention_20(conv1)
        return attention_weights

    # def _init_attention(self, h3):
    #     trainable = True
    #     if self.hparams.get('attention_frozen'): trainable = False
    #     init_scale = 0.0 if self.hparams.get('init_attention_zero') else 1.0

    #     attention_logits, attention_selector = conv_without_bias(h3, 'attention_logits', nf=1, rf=1, stride=1, init_scale=init_scale, trainable=trainable)

    #     if self.hparams['attention_cosine_distance']:
    #         attention_selector_norm = tf.norm(tf.reshape(attention_selector, [-1]))
    #         # Was getting nan issues with tf.norm, manually implementing it seemed to work though?
    #         h3_norm = tf.sqrt(tf.reduce_sum(tf.square(h3), axis=-1, keep_dims=True) + 1e-7)
    #         attention_logits /= (attention_selector_norm * h3_norm + 1e-5)

    #     self.attention_logits = attention_logits

    #     orig_attention_shape = attention_logits.get_shape()
    #     attention_logits = tf.reshape(attention_logits, (self.nbatch, -1))
    #     attention_weights = tf.nn.softmax(attention_logits / self.hparams['attention_temperature'])
    #     attention_weights = tf.reshape(attention_weights, orig_attention_shape)
    #     self.attention_weights = attention_weights

    #     if self.reuse:
    #         tf.summary.image('predicted_attention', self.attention_weights, max_outputs=1)

    # def _get_noisy_pi_and_vf(self, reuse, init_scales):
    #     random_dim = 64 if self.hparams.get('dropout_more_random') else 1
    #     random_in = tf.truncated_normal(shape=[80, 7, 7, random_dim])
    #     stddev = tf.sqrt((1.0 - self.attention_weights) / (self.attention_weights + 1e-6) + 1e-7)
    #     stddev *= self.hparams['_dropout_strength']
    #     noise = self.h3 * (random_in * stddev)

    #     if reuse:
    #         tf.summary.scalar('noise_min', tf.reduce_min(tf.abs(noise)))
    #         tf.summary.scalar('noise_max', tf.reduce_max(tf.abs(noise)))
    #         tf.summary.scalar('noise_uncentered_mean', tf.reduce_mean(tf.abs(noise)))

    #         tf.summary.scalar('h3_min', tf.reduce_min(tf.abs(self.h3)))
    #         tf.summary.scalar('h3_max', tf.reduce_max(tf.abs(self.h3)))
    #         tf.summary.scalar('h3_uncentered_mean', tf.reduce_mean(tf.abs(self.h3)))

    #     h3_noise = self.h3 + noise
    #     h3_noise = conv_to_fc(h3_noise)

    #     h_noise = tf.nn.relu(fc(h3_noise, 'fc1', nh=512, init_scale=np.sqrt(2), trainable=True, reuse=True))
    #     pi_noise = fc(h_noise, 'pi', self.nact, init_scale=init_scales[0], reuse=True)
    #     vf_noise = fc(h_noise, 'v', 1, init_scale=init_scales[1], reuse=True)[:,0]
    #     return pi_noise, vf_noise

    def __init__(self, sess, X, ob_space, ac_space, nbatch, nsteps, reuse=False, hparams=None): #pylint: disable=W0613
        assert hparams != None
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        self.X = X
        self.nbatch = nbatch
        self.reuse = reuse
        self.hparams = hparams
        self.nact = nact

        scope = hparams.get('_policy_scope', 'model')

        with tf.variable_scope(scope, reuse=reuse):
            # if '_teacher_h3' in hparams:
            #     h3 = self.hparams['_teacher_h3']
            # else:
            #     h3, _h3 = nature_cnn_h3(X, 
            #         first_layer_mode=hparams['first_layer_mode'], 
            #         trainable=hparams['base_trainable'],
            #         conv1_fn=lambda x: self._init_attention_20(x))

            h3, _h3 = nature_cnn_h3(X, 
                first_layer_mode=hparams['first_layer_mode'], 
                trainable=hparams['base_trainable'],
                conv1_fn=lambda x: self._init_attention_20_main_branch(x))

            # self.flow_base = h3

            # self.attention_weights = tf.ones(shape=[nbatch, 7, 7, 1], dtype=tf.float32) / 49.0

            # if '_attention_truth' in hparams:
            #     self.attention_weights = tf.reshape(self.hparams['_attention_truth'], [nbatch, 7, 7, 1])
            # else:
            #     self._init_attention(_h3)

            # attention_weighted_vectors = tf.reduce_sum(h3 * self.attention_weights, axis=[1,2])

            # # Sanity check hparams.
            # attention_modes = ['use_global_vecs', 'attention_weighted_fc']
            # mode_count = sum(hparams.get(mode) is True for mode in attention_modes)
            # assert mode_count <= 1


            # if hparams.get('use_global_vecs'):
            #     global_vecs = tf.nn.relu(conv(h3, 'global_vec_conv', nf=4, rf=1, stride=1, trainable=True))
            #     global_vecs = conv_to_fc(global_vecs)

            #     h = tf.concat([attention_weighted_vectors, global_vecs], -1)
            #     assert len(h.get_shape()) == 2
            #     assert h.get_shape()[-1] == 260

            #     for i in range(hparams['fc_depth']):
            #         h = tf.nn.relu(fc(h, 'additional_fc{}'.format(i), nh=260, init_scale=np.sqrt(2)))
            # elif hparams.get('attention_weighted_fc'):
            #     # attention_entropy = cat_entropy(tf.reshape(self.attention_logits, [-1, 49]))
            #     # max_entropy = np.log(49)
            #     # entropy_percentage = attention_entropy / max_entropy
            #     # attention_scaling_factor = 49.0 * entropy_percentage

            #     # if reuse:
            #     #     tf.summary.scalar('max_attention_scaling_factor', tf.reduce_max(attention_scaling_factor))
            #     #     tf.summary.scalar('min_attention_scaling_factor', tf.reduce_min(attention_scaling_factor))
            #     #     tf.summary.scalar('mean_attention_scaling_factor', tf.reduce_mean(attention_scaling_factor))

            #     # attention_scaling_factor = tf.expand_dims(tf.expand_dims(tf.expand_dims(attention_scaling_factor, axis=1), axis=1), axis=1)
            #     # attention_scaling_factor = tf.stop_gradient(attention_scaling_factor)

            #     # Intentionally not using the attention weighted vectors in this path.
            #     h3 = h3 * self.attention_weights * 49.0 # Multiply so that we don't change the scale too much, attention=1/49 if uniform
            #     self.h3 = h3
            #     h3_flat = conv_to_fc(h3)
            #     h = tf.nn.relu(fc(h3_flat, 'fc1', nh=512, init_scale=np.sqrt(2), trainable=True))

            #     if hparams.get('add_global_vecs'):
            #         h = tf.concat([h, attention_weighted_vectors], axis=-1)
            #         assert h.get_shape()[-1] == 512 + 64

            #     #for i in range(hparams['fc_depth']):
            #     #    h = tf.nn.relu(fc(h, 'additional_fc{}'.format(i), nh=512, init_scale=np.sqrt(2)))
            # elif hparams.get('gaussian_attention'):
            #     # Gaussian dropout + reparamaterization trick from vae's.
            #     if self.hparams.get('fixed_dropout_noise'):
            #         random_in = hparams['_env_random']
            #     else:
            #         random_dim = 64 if hparams.get('dropout_more_random') else 1
            #         random_in = tf.truncated_normal(shape=[7, 7, random_dim])
            #     stddev = tf.sqrt((1.0 - self.attention_weights) / (self.attention_weights + 1e-6) + 1e-7)
            #     stddev *= hparams['_dropout_strength']
            #     noise = h3 * (random_in * stddev)

            #     if reuse:
            #         tf.summary.scalar('noise_min', tf.reduce_min(tf.abs(noise)))
            #         tf.summary.scalar('noise_max', tf.reduce_max(tf.abs(noise)))
            #         tf.summary.scalar('noise_uncentered_mean', tf.reduce_mean(tf.abs(noise)))

            #         tf.summary.scalar('h3_min', tf.reduce_min(tf.abs(h3)))
            #         tf.summary.scalar('h3_max', tf.reduce_max(tf.abs(h3)))
            #         tf.summary.scalar('h3_uncentered_mean', tf.reduce_mean(tf.abs(h3)))

            #     h3 = h3 + noise

            #     h3 = conv_to_fc(h3)
            #     h = tf.nn.relu(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), trainable=True))

            #     for i in range(hparams['fc_depth']):
            #         h = tf.nn.relu(fc(h, 'additional_fc{}'.format(i), nh=512, init_scale=np.sqrt(2)))
            # else:
            #     h3 = conv_to_fc(h3)
            #     h = tf.nn.relu(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), trainable=True))
            #     h = tf.concat([attention_weighted_vectors, h], -1)
            #     assert len(h.get_shape()) == 2
            #     assert h.get_shape()[-1] == (512+64)

            #     for i in range(hparams['fc_depth']):
            #         h = tf.nn.relu(fc(h, 'additional_fc{}'.format(i), nh=(512+64), init_scale=np.sqrt(2)))


            h3 = conv_to_fc(h3)
            h = tf.nn.relu(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), trainable=True))
            self.original_h = h

            if hparams['use_extra_path']:
                with tf.variable_scope('model_extrapath', reuse=reuse):

                    h3, _h3 = nature_cnn_h3(X, 
                        first_layer_mode=hparams['first_layer_mode'], 
                        trainable=hparams['base_trainable'],
                        conv1_fn=lambda x: self._init_attention_20_extrapath(x))

                    h3 = conv_to_fc(h3)
                    notransfer_h = tf.nn.relu(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), trainable=True))
                    print("notransfer_h: {}".format(notransfer_h))
                    print('original h: {}'.format(h))

                    concatenated_h = tf.concat([h, notransfer_h], axis=-1)
                    print('concatenated_h: {}'.format(concatenated_h))

                    h = tf.nn.relu(fc(concatenated_h, 'extra_path_fc', nh=512, init_scale=np.sqrt(2), trainable=True))
                    print('final h: {}'.format(h))

            self.h = h

            if hparams['context_20']:
                h = tf.concat([h, self.context_20], axis=-1)

            init_scales = [0.01, 1]

            # if hparams.get('init_pi_v_zero'):
            #     init_scales = [0 for _ in init_scales]

            pi = fc(h, 'pi', nact, init_scale=init_scales[0])
            vf = fc(h, 'v', 1, init_scale=init_scales[1])[:,0]

            if reuse:
                tf.summary.scalar('vf', tf.reduce_mean(vf))


            # if hparams.get('gaussian_attention') and hparams.get('attention_weighted_fc') and reuse:
            #     self.pi_noises = []
            #     self.vf_noises = []

            #     for i in range(hparams.get("num_dropout_models")):
            #         pi_noise, vf_noise = self._get_noisy_pi_and_vf(reuse, init_scales)
            #         self.pi_noises.append(pi_noise)
            #         self.vf_noises.append(vf_noise)

                # self.pi_noise = pi_noise
                # self.vf_noise = vf_noise

                # pi_target = tf.stop_gradient(tf.nn.softmax(pi))
                # vf_target = tf.stop_gradient(vf)

                # noise_pi_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pi_noise, labels=pi_target)
                # noise_vf_loss = hparams['_vf_coef'] * mse(vf_noise, vf_target)
                # noise_loss = tf.reduce_mean(noise_pi_loss+noise_vf_loss) * hparams['noise_loss_c']

                # if reuse:
                #     tf.summary.scalar('noise_pi_loss', tf.reduce_mean(noise_pi_loss))
                #     tf.summary.scalar('noise_vf_loss', tf.reduce_mean(noise_vf_loss))
                #     tf.summary.scalar('noise_loss', noise_loss)

                # self.noise_loss = noise_loss


        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        # reshaped_attn = tf.reshape(self.attention_weights_20, [-1, 400])

        def step(ob, *_args, **_kwargs):
            feed_dict = {X:ob}
            if '_dropout_strength' in _kwargs:
                (k, v) = _kwargs['_dropout_strength']
                feed_dict[k] = v

            ops = [a0, vf, neglogp0]

            results = sess.run(ops, feed_dict)
            return results[0], results[1], results[2]

        def value(ob, *_args, **_kwargs):
            feed_dict = {X:ob}
            if '_dropout_strength' in _kwargs:
                (k, v) = _kwargs['_dropout_strength']
                feed_dict[k] = v
            
            return sess.run(vf, feed_dict)

        # copy_weights_op = None
        # def setup_copy_weights():
        #     nonlocal copy_weights_op
        #     assert hparams.get('_src_scope')

        #     layer_names = ['c1_2_frame_input', 'c2', 'c3', 'fc1', 'pi', 'v', 'attention_logits']
        #     layer_types = ['w', 'b']
        #     assert hparams['fc_depth'] == 0

        #     src_scope = hparams['_src_scope']
        #     assign_ops = []

        #     for layer_name in layer_names:
        #         for layer_type in layer_types:
        #             if layer_name == 'attention_logits' and layer_type == 'b': continue

        #             var_name = layer_name + '/' + layer_type + ':0'
        #             dst_var_name = scope + '/' + var_name
        #             src_var_name = src_scope + '/' + var_name

        #             dst_var = None
        #             src_var = None

        #             for v in tf.global_variables():
        #                 if v.name == dst_var_name: dst_var = v
        #                 if v.name == src_var_name: src_var = v

        #             assert dst_var is not None
        #             assert src_var is not None

        #             assign_op = tf.assign(ref=dst_var, value=src_var)
        #             assign_ops.append(assign_op)

        #     copy_weights_op = tf.group(*assign_ops)

        # def copy_weights():
        #     sess.run(copy_weights_op)
        # self.copy_weights = copy_weights
        # self.setup_copy_weights = setup_copy_weights

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
