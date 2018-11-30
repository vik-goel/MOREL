import os
import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse, fc, conv, sample, cat_entropy_softmax

from baselines.common.cmd_util import publish_attention_weights


import unsupervised_object_segmentation.object_segmentation_network as object_segmentation
import copy

_all_cnn_var_names = set([
    'model/c1_2_frame_input/w:0',
    'model/c1_2_frame_input/b:0',
    'model/c1/w:0',
    'model/c1/b:0',
    'model/c2/w:0',
    'model/c2/b:0',
    'model/c3/w:0',
    'model/c3/b:0',
    'model/fc1/w:0',
    'model/fc1/b:0',
])

def do_restore(sess, saver, ckpt_dir):
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    assert ckpt_path is not None
    saver.restore(sess, ckpt_path)

def get_dropout_strength(hparams, step):
    if hparams.get('dropout_strength_warm_up'):
        return min(hparams['dropout_strength'], float(step) / hparams['dropout_strength_warm_up'] * hparams['dropout_strength'])
    else:
        return hparams['dropout_strength']


def restore_teacher_from_checkpoint(sess, ckpt_path):
    var_names = [
        'teacher/c1_2_frame_input/w',
        'teacher/c1_2_frame_input/b',

        'teacher/c1/w',
        'teacher/c1/b',
        'teacher/c2/w',
        'teacher/c2/b',
        'teacher/c3/w',
        'teacher/c3/b',
        'teacher/fc1/w',
        'teacher/fc1/b',

        'teacher/fc2/w',
        'teacher/fc2/b',

        'teacher/conv4/w',
        'teacher/conv4/b',
        'teacher/conv5/w',
        'teacher/conv5/b',
        'teacher/conv6/w',
        'teacher/conv6/b',
        'teacher/masks/w',
        'teacher/masks/b',

        'teacher/flow/camera_translation/w',
        'teacher/flow/camera_translation/b',

        'teacher/object_translation/obj_t/w',
        'teacher/object_translation/obj_t/b',

        'teacher/upsample1/conv4/w',
        'teacher/upsample1/conv4/b',
        'teacher/upsample2/conv5/w',
        'teacher/upsample2/conv5/b',
        'teacher/upsample3/conv6/w',
        'teacher/upsample3/conv6/b',
        'teacher/upsample4/conv7/w',
        'teacher/upsample4/conv7/b',

        'teacher/object_masks/masks/w',
        'teacher/object_masks/masks/b',
    ]

    var_dict = {}
    for var_name in var_names:
        var_options = [v for v in tf.global_variables() if v.name == var_name+':0']
        if var_options:
            print('restored {}'.format(var_name))
            var_dict[var_name.replace('teacher', 'model')] = var_options[0]
        else:
            print('could not restore {}'.format(var_name))

    saver = tf.train.Saver(var_list=var_dict)
    do_restore(sess, saver, ckpt_path)



class Model(object):

    def _create_object_segmentation_net(self, X, reuse=False, is_step_model=False, embedding=None):
        assert self.hparams.get('teacher_ckpt')

        scaled_images = tf.cast(X, dtype=tf.float32) / 255.

        # Create a sfm_base.
        frame0 = tf.expand_dims(scaled_images[..., -2], axis=-1)
        frame1 = tf.expand_dims(scaled_images[..., -1], axis=-1)
        print('frame0_shape: {}'.format(frame0.get_shape()))
        print('frame1_shape: {}'.format(frame1.get_shape()))

        sfm_base = object_segmentation.ObjectSegmentationBase(frames=scaled_images[..., -2:], embedding=embedding)

        sfm_hparams = copy.deepcopy(self.hparams)
        sfm_hparams['batch_size'] = self.nenvs*self.nsteps

        result = object_segmentation.ObjectSegmentationNet(
            hparams=sfm_hparams,
            sfm_base=sfm_base,
            is_teacher_network=True,
            reuse=reuse,
            is_step_model=is_step_model,
            trainable=self.hparams['do_joint_training']
        )

        return result

    def _object_masks_to_attention(self, object_masks, object_translation):
        use_mask = tf.cast(tf.reduce_sum(object_masks, axis=[1, 2]) > 2.0, tf.float32)
        teach_mask = tf.cast(tf.reduce_sum(use_mask, axis=-1) > 0, tf.float32)
        object_masks *= tf.expand_dims(tf.expand_dims(use_mask, axis=1), axis=1)

        reduce_op = tf.reduce_max if self.hparams.get('max_attention_truth') else tf.reduce_sum

        img_w = self.hparams['img_w']
        img_h = self.hparams['img_h']
        batch_size = self.hparams['batch_size']
        k_obj = self.hparams['k_obj']

        assert object_masks.get_shape()[0] == batch_size
        assert object_masks.get_shape()[3] == k_obj
        assert len(object_masks.get_shape()) == 4

        if self.hparams.get('small_object_prior'):
            border = 2
            cutoff = 0.1
            num_masks = batch_size * k_obj

            object_masks_flat = tf.transpose(object_masks, [0, 3, 1, 2])
            object_masks_flat = tf.reshape(object_masks_flat, [num_masks, img_h, img_w])

            border_mask = np.ones([84,84])
            for i in range(border):
                border_mask[i,:] = 0
                border_mask[:,i] = 0

            mask = object_masks_flat * border_mask
            mask = (mask - cutoff) / (1. - cutoff)
            mask *= tf.cast(mask > 0, tf.float32)
            mask = tf.reshape(mask, [num_masks, -1])
            mask_sum = tf.reduce_sum(mask, axis=-1)

            # Assume mask_sum > epsilon.
            mask /= tf.expand_dims(mask_sum, axis=-1)

            x_linspace = tf.linspace(0., img_w - 1., img_w)
            y_linspace = tf.linspace(0., img_h - 1., img_h)
            x_coord, y_coord = tf.meshgrid(x_linspace, y_linspace)

            x_coord = tf.reshape(x_coord, [1, -1])
            y_coord = tf.reshape(y_coord, [1, -1])
            mesh_grid = tf.concat([y_coord, x_coord], axis=0)
            mesh_grid = tf.expand_dims(tf.transpose(mesh_grid), axis=0)

            mean_coord = tf.reduce_sum(mesh_grid * tf.expand_dims(mask, -1), axis=1)
            mean_coord = tf.expand_dims(mean_coord, 1)
            assert(mean_coord.get_shape() == [num_masks, 1, 2])

            diffs = mask * tf.reduce_sum(tf.square(mesh_grid - mean_coord), axis=-1)
            diffs = tf.reduce_mean(diffs, axis=-1)

            weight = (1.0 / diffs)
            weight = tf.where(condition=tf.is_nan(weight), x=tf.zeros_like(weight), y=weight)
            weight = tf.where(condition=tf.is_inf(weight), x=tf.zeros_like(weight), y=weight)
            weight = tf.reshape(weight, [batch_size, 1, 1, k_obj])
        elif self.hparams.get('object_flow_weighting'):
            flow_norms = tf.reduce_sum(tf.abs(object_translation), axis=-1) + 1e-6
            flow_totals = tf.reduce_sum(flow_norms, axis=-1, keep_dims=True)
            weight = tf.expand_dims(tf.expand_dims(flow_norms / flow_totals, axis=1), axis=1)
        else:
            weight = np.ones([batch_size, 1, 1, k_obj])

        summed_masks = reduce_op(object_masks * weight, axis=-1, keep_dims=True)
        return summed_masks, teach_mask

    def _get_attention_truth(self, teacher, is_step_model):
        # Forward pass the teacher to get object masks.
        # Sum the object masks to get all the object pixels across all the masks.
        summed_masks, teach_mask = self._object_masks_to_attention(teacher.object_masks, teacher.object_translation)

        if not is_step_model:
            tf.summary.image('summed_masks', summed_masks, max_outputs=1)

        if self.hparams.get('correct_attention_truth'):
            coarse_masks = tf.nn.conv2d(summed_masks, filter=np.ones([8,8,1,1],dtype=np.float32), strides=[1,4,4,1], padding='VALID')
            coarse_masks = tf.nn.conv2d(coarse_masks, filter=np.ones([4,4,1,1],dtype=np.float32), strides=[1,2,2,1], padding='VALID')
            coarse_masks = tf.nn.conv2d(coarse_masks, filter=np.ones([3,3,1,1],dtype=np.float32), strides=[1,1,1,1], padding='VALID')
        elif self.hparams.get('resize_attention_downsampling'):
            coarse_masks = tf.image.resize_images(summed_masks, size=[7, 7], method=tf.image.ResizeMethod.AREA)
        else:
            # Make them coarser by summing the entire 12x12 region. (eg. downsample from 84x84 to 7x7 by summing in 12x12 blocks)
            coarse_masks = tf.nn.conv2d(summed_masks, filter=np.ones([12,12,1,1],dtype=np.float32), strides=[1,12,12,1], padding='SAME')

        batch_size = self.nenvs if is_step_model else self.nbatch

        coarse_masks = tf.reshape(coarse_masks, [batch_size, -1]) + 1e-6
        print('coarse_masks: {}'.format(coarse_masks.get_shape()))


        #attention_truth = tf.nn.softmax(coarse_masks / hparams['attention_truth_temperature'])
        attention_truth = coarse_masks / tf.reduce_sum(coarse_masks, axis=-1, keep_dims=True)

        print('attention_truth: {}'.format(attention_truth.get_shape()))

        # Do not backprop into the labels and try to train the teacher -- the student is not the master.
        attention_truth = tf.stop_gradient(attention_truth)
        return attention_truth, teach_mask

    def _get_attention_truth_20(self, teacher, is_step_model):
        # Forward pass the teacher to get object masks.
        # Sum the object masks to get all the object pixels across all the masks.
        summed_masks, teach_mask = self._object_masks_to_attention(teacher.object_masks, teacher.object_translation)

        #if not is_step_model:
        #    tf.summary.image('summed_masks', summed_masks, max_outputs=1)

        # Make them coarser by summing the entire 12x12 region. (eg. downsample from 84x84 to 7x7 by summing in 12x12 blocks)
        if self.hparams.get('resize_attention_downsampling'):
            coarse_masks = tf.image.resize_images(summed_masks, size=[20, 20], method=tf.image.ResizeMethod.AREA)
        else:
            coarse_masks = tf.nn.conv2d(summed_masks, filter=np.ones([8,8,1,1], dtype=np.float32), strides=[1,4,4,1], padding='VALID')

        batch_size = self.nenvs if is_step_model else self.nbatch

        coarse_masks = tf.reshape(coarse_masks, [batch_size, -1]) + self.hparams['mask_smoothing']
        print('coarse_masks_20: {}'.format(coarse_masks.get_shape()))

        # attention_truth = tf.nn.softmax(coarse_masks / hparams['attention_truth_temperature'])
        attention_truth = coarse_masks / tf.reduce_sum(coarse_masks, axis=-1, keep_dims=True)
        print('attention_truth_20: {}'.format(attention_truth.get_shape()))


        # Do not backprop into the labels and try to train the teacher -- the student is not the master.
        attention_truth = tf.stop_gradient(attention_truth)
        return attention_truth, teach_mask


    # def _get_flow_truth(self, teacher):
    #     flow = tf.transpose(teacher.flow, [0, 2, 3, 1])
    #     assert flow.get_shape()[-1] == 2

    #     filter_weights = np.zeros([12, 12, 2, 2], dtype=np.float32)
    #     filter_weights[:, :, 0, 0] = 1.0 / 144.0
    #     filter_weights[:, :, 1, 1] = 1.0 / 144.0

    #     coarse_flow = tf.nn.conv2d(flow, filter=filter_weights, strides=[1,12,12,1], padding='SAME')

    #     tf.summary.image('truth_flow_x', tf.expand_dims(coarse_flow[..., 0], axis=-1), max_outputs=1)
    #     tf.summary.image('truth_flow_y', tf.expand_dims(coarse_flow[..., 1], axis=-1), max_outputs=1)

    #     x_labels = tf.cast(coarse_flow[..., 0] > 0, tf.int32)
    #     y_labels = tf.cast(coarse_flow[..., 1] > 0, tf.int32)

    #     x_labels = tf.reshape(x_labels, [-1])
    #     y_labels = tf.reshape(y_labels, [-1])

    #     return tf.stop_gradient(x_labels), tf.stop_gradient(y_labels)



    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            hparams=None):
        assert hparams != None
        hparams['_vf_coef'] = vf_coef

        # Create the session.
        sess = tf_util.make_session(per_process_gpu_memory_fraction=hparams.get('gpu_fraction', 0.25))
        self.sess = sess

        # Copy hparams.
        self.hparams = hparams
        self.nenvs = nenvs
        self.nsteps = nsteps

        self.hparams['batch_size'] = nenvs*nsteps

        # Setup constants.
        nact = ac_space.n
        nbatch = nenvs*nsteps
        self.nbatch = nbatch
        nh, nw, nc = ob_space.shape
        ob_shape_train = (nbatch, nh, nw, nc)
        ob_shape_step = (nenvs, nh, nw, nc)

        # Setup placeholders.
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        TEACHER_C = tf.placeholder(tf.float32, [])
        DROPOUT_STRENGTH = tf.placeholder(tf.float32, [], name='DROPOUT_STRENGTH')
        self.DROPOUT_STRENGTH = DROPOUT_STRENGTH
        X_train = tf.placeholder(tf.float32, ob_shape_train, name='Ob_train') #obs
        X_step = tf.placeholder(tf.float32, ob_shape_step, name='Ob_step') #obs
        attention_truth = None

        step_hparams = copy.deepcopy(hparams)
        train_hparams = copy.deepcopy(hparams)

        # if self.hparams.get('fixed_dropout_noise'):
        #     self.step_env_random = tf.get_variable(
        #         shape=[nenvs, 7, 7, 1],
        #         name='env_random',
        #         initializer=tf.truncated_normal_initializer(),
        #         trainable=False,
        #     )

        #     self.train_env_random = tf.tile(tf.expand_dims(self.step_env_random, axis=0), multiples=[nsteps, 1, 1, 1, 1])
        #     self.train_env_random = tf.reshape(
        #         tf.transpose(self.train_env_random, perm=[1, 0, 2, 3, 4]),
        #         [nbatch, 7, 7, 1])

        #     step_hparams['_env_random'] = self.step_env_random
        #     train_hparams['_env_random'] = self.train_env_random

        # train_hparams['_dropout_strength'] = DROPOUT_STRENGTH
        # step_hparams['_dropout_strength'] = DROPOUT_STRENGTH

        # Create the models.
        step_model = policy(sess, X_step, ob_space, ac_space, nenvs, 1, reuse=False, hparams=step_hparams)
        train_model = policy(sess, X_train, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True, hparams=train_hparams)

        if hparams.get('teacher_ckpt'):
            assert hparams.get('use_fixed_attention') or hparams.get('learn_attention_from_teacher') or hparams.get('do_joint_training')

            # Create the teacher, so that way we can use its attention weights
            # instead of learning how to do attention on our own.
            # step_teacher = self._create_sfmnet(X_step, reuse=False, is_step_model=True)

            train_teacher = self._create_object_segmentation_net(
                X_train,
                reuse=False,
                is_step_model=False,
                embedding=train_model.original_h if hparams['do_joint_training'] else None,
            )
            train_attention_truth, train_attention_mask = self._get_attention_truth(train_teacher, is_step_model=False)

            # step_attention_truth = self._get_attention_truth(step_teacher, is_step_model=True)

            # if hparams.get('use_fixed_attention'):
            #     step_hparams['_attention_truth'] = step_attention_truth
            #     train_hparams['_attention_truth'] = train_attention_truth

            # if hparams.get('do_joint_training'):
            #     step_hparams['_teacher_h3'] = step_teacher.conv3
            #     step_hparams['_teacher_h'] = step_teacher.embedding

            #     train_hparams['_teacher_h3'] = train_teacher.conv3
            #     train_hparams['_teacher_h'] = train_teacher.embedding

        # if hparams.get('use_target_model'):
        #     assert not hparams.get('do_joint_training')

        #     target_hparams = copy.copy(train_hparams)
        #     target_hparams['_policy_scope'] = 'target_model'
        #     target_hparams['_src_scope'] = 'model'
        #     target_model = policy(sess, X_step, ob_space, ac_space, nenvs, 1, reuse=False, hparams=target_hparams)
        #     target_model.setup_copy_weights()
        #     self.target_model = target_model

        scaled_images = tf.cast(train_model.X, tf.float32) / 255.
        print('scaled_images shape: {}'.format(scaled_images))

        sfm_base = object_segmentation.ObjectSegmentationBase(frames=scaled_images, embedding=train_model.h)
        sfm_hparams = copy.deepcopy(hparams)
        sfm_hparams['batch_size'] = nenvs*nsteps

        tf.summary.image('frame0', tf.expand_dims(train_model.X[..., -2], axis=-1), max_outputs=1)
        tf.summary.image('frame1', tf.expand_dims(train_model.X[..., -1], axis=-1),  max_outputs=1)

        # Create the loss function.
        def a2c_loss(pi, vf):
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
            pg_loss = tf.reduce_mean(ADV * neglogpac)
            vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
            entropy = tf.reduce_mean(cat_entropy(train_model.pi))

            # ent_coef_mode = hparams.get('ent_coef_mode', 'default')
            # ent_coef_val = hparams.get('ent_coef_val', ent_coef)

            # if ent_coef_mode == 'default':
            #     actual_ent_coef = ent_coef_val
            # elif ent_coef_mode == 'linear_teacher':
            #     actual_ent_coef = ent_coef_val * TEACHER_C + ent_coef * (1 - TEACHER_C)
            # elif ent_coef_mode == 'additive_teacher':
            #     actual_ent_coef = ent_coef_val + ent_coef_val * TEACHER_C
            # else:
            #     raise Exception('unrecognized ent_coef_mode: {}'.format(ent_coef_mode))

            loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef
            return loss, pg_loss, vf_loss, entropy

        loss, pg_loss, vf_loss, entropy = a2c_loss(train_model.pi, train_model.vf)

        # if hparams.get('dropout_data_aug_c'):
        #     logged_augs = False
        #     loss_c = 1.0 - hparams['num_dropout_models'] * hparams['dropout_data_aug_c']
        #     assert loss_c >= hparams['dropout_data_aug_c'] - 1e-5
        #     loss = loss_c * loss

        #     for pi_noise, vf_noise in zip(train_model.pi_noises, train_model.vf_noises):
        #         l2, pg2, vf2, entropy2 = a2c_loss(pi_noise, vf_noise)
        #         loss += l2 * hparams['dropout_data_aug_c']

        #         if not logged_augs:
        #             logged_augs = True
        #             tf.summary.scalar('aug_loss', tf.reduce_mean(l2))
        #             tf.summary.scalar('aug_pgloss', tf.reduce_mean(pg2))
        #             tf.summary.scalar('aug_vfloss', tf.reduce_mean(vf2))
        #             tf.summary.scalar('aug_entropyloss', tf.reduce_mean(entropy2))

        #     print("ADDING DROPOUT DATA AUG")

        # if hasattr(train_model, 'noise_loss') and hparams.get('noise_loss_c'):
        #     loss += train_model.noise_loss
        #     print("ADDING NOISE LOSS")

        # tf.summary.image('frame0', tf.expand_dims(train_model.X[..., -2],-1), max_outputs=1)
        # tf.summary.image('frame1', tf.expand_dims(train_model.X[..., -1],-1),  max_outputs=1)

        teacher_loss = 0.0

        if hparams.get('teacher_ckpt') and hparams.get('learn_attention_from_teacher'):
            assert hparams.get('attention_20') or hparams.get('inverted_attention_20')
            # Load in the teacher.
            # teacher = sfmnet.SfmNet(hparams=sfm_hparams, sfm_base=sfm_base, is_teacher_network=True)

            # attention_loss = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=train_attention_truth,
            #     logits=tf.reshape(train_model.attention_logits, [nbatch,-1])
            # )
            # print('attention_loss: {}'.format(attention_loss.get_shape()))
            # print('train_attention_mask: {}'.format(train_attention_mask.get_shape()))
            # attention_loss = attention_loss * train_attention_mask
            # attention_loss = tf.reduce_mean(attention_loss)

            # # for t in [5., 10., 20., 40., 75., 100., 200., 500., 1000.]:
            # #     truth = tf.nn.softmax(coarse_masks / t)
            # #     tf.summary.image('attention_truth_{}'.format(t), tf.reshape(truth, [nbatch, 7, 7, 1]), max_outputs=1)
            # tf.summary.scalar('attention_loss', attention_loss)
            # tf.summary.scalar('attention_teaching', tf.reduce_mean(train_attention_mask))

            # teacher_loss = TEACHER_C * attention_loss

            tf.summary.scalar('teacher_c', TEACHER_C)
            truth, mask = self._get_attention_truth_20(train_teacher, is_step_model=False)
            tf.summary.image('attention_20_truth', tf.reshape(truth, [80, 20, 20, 1]), max_outputs=1)

            if hparams.get('attention_20'):
                attention_loss_20 = tf.nn.softmax_cross_entropy_with_logits(
                    labels=truth,
                    logits=tf.reshape(train_model.attention_logits_20, [-1, 400])
                )
                attention_loss_20 = tf.reduce_mean(attention_loss_20 * mask)

                tf.summary.scalar('attention_loss_20', attention_loss_20)
                tf.summary.scalar('attention_teaching_20', tf.reduce_mean(mask))
                teacher_loss += TEACHER_C * attention_loss_20

            if hparams.get('extrapath_attention_20'):
                print("EXTRAPATH ATTENTION!!!")
                attention_loss_20 = tf.nn.softmax_cross_entropy_with_logits(
                    labels=truth,
                    logits=tf.reshape(train_model.extrapath_attention_logits_20, [-1, 400])
                )
                attention_loss_20 = tf.reduce_mean(attention_loss_20 * mask)

                tf.summary.scalar('attention_loss_20', attention_loss_20)
                tf.summary.scalar('attention_teaching_20', tf.reduce_mean(mask))
                teacher_loss += (-TEACHER_C) * attention_loss_20


        # if hparams.get('learn_attention_from_pg'):
        #     attention_logits = tf.reshape(train_model.attention_logits, [nbatch, 49])
        #     attention_actions = sample(attention_logits)
        #     attention_actions = tf.stop_gradient(attention_actions)

        #     attention_neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=attention_logits, labels=attention_actions)
        #     attention_pg_loss = tf.reduce_mean(ADV * attention_neglogpac)

        #     tf.summary.scalar('attention_pg_loss', attention_pg_loss)

        #     loss += attention_pg_loss * hparams['learn_attention_from_pg']

        # if hparams.get('teacher_ckpt') and hparams.get('learn_translation_from_teacher'):
        #     with tf.variable_scope("model"):
        #         with tf.variable_scope('object_translation'):
        #             pred_translation = fc(train_model.h, 'obj_t', nh=2*self.hparams['k_obj'], init_scale=1.0)
        #             pred_translation = tf.reshape(pred_translation, (-1, self.hparams['k_obj'], 2))

        #     teacher_translation = tf.stop_gradient(train_teacher.object_translation)
        #     translation_loss = mse(pred_translation, teacher_translation)
        #     translation_loss = tf.reduce_mean(translation_loss)
        #     teacher_loss += TEACHER_C * translation_loss
        #     tf.summary.scalar('translation_loss', translation_loss)

        if hparams['do_joint_training']:
            teacher_loss += tf.reduce_mean(train_teacher.transform_loss + train_teacher.mask_reg_loss) * TEACHER_C

        if hasattr(train_model, 'attention_logits_20'):
            # Want a low entropy distribution, so that we are focused on only a small part of the image per frame.
            reshaped_logits = tf.reshape(train_model.attention_logits_20, [-1, 400])
            attention_entropy = tf.reduce_mean(cat_entropy(reshaped_logits))
            teacher_loss -= hparams['attention_entropy_c'] * attention_entropy * TEACHER_C

            tf.summary.scalar('attention_entropy', attention_entropy)

        if hasattr(train_model, 'extrapath_attention_logits_20'):
            # Want a low entropy distribution, so that we are focused on only a small part of the image per frame.
            reshaped_logits = tf.reshape(train_model.extrapath_attention_logits_20, [-1, 400])
            attention_entropy = tf.reduce_mean(cat_entropy(reshaped_logits))
            teacher_loss -= hparams['attention_entropy_c'] * attention_entropy * TEACHER_C

            tf.summary.scalar('extrapath_attention_entropy', attention_entropy)


        # if hasattr(train_model, 'attention_weights_20'):
        #     # Want this to be high entropy, so we are looking at different parts of the image on different images.
        #     batch_logits = tf.reshape(tf.reduce_sum(train_model.attention_weights_20, axis=0), [1, 400])
        #     attention_entropy = tf.reduce_mean(cat_entropy_softmax(batch_logits))
        #     loss -= hparams['batch_entropy_c'] * attention_entropy
        #     tf.summary.scalar('batch_entropy', attention_entropy)

        # if hparams['do_joint_training'] and False:
        #     assert hparams.get('teacher_ckpt')
        #     teacher_loss += TEACHER_C * train_teacher.total_loss
        # else:
        #     sfm_loss = None

        # if hparams['do_flow_prediction']:
        #     assert hparams.get('teacher_ckpt')
        #     flow_truth_x, flow_truth_y = self._get_flow_truth(train_teacher)
        #     predicted_flow = conv(train_model.flow_base, 'pred_flow', nf=4, rf=1, stride=1, trainable=True)

        #     flow_pred_x = tf.reshape(predicted_flow[..., :2], [-1, 2])
        #     flow_pred_y = tf.reshape(predicted_flow[..., 2:], [-1, 2])

        #     flow_x_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flow_truth_x, logits=flow_pred_x))
        #     flow_y_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flow_truth_y, logits=flow_pred_y))
        #     flow_loss = flow_x_loss + flow_y_loss

        #     # flow_error = tf.reduce_mean(mse(flow_truth, predicted_flow))
        #     teacher_loss += TEACHER_C * flow_loss * hparams['flow_error_c']

        #     flow_x_acc = tf.reduce_mean(tf.cast(tf.argmax(flow_pred_x, axis=-1) == flow_truth_x, tf.int32))
        #     flow_y_acc = tf.reduce_mean(tf.cast(tf.argmax(flow_pred_y, axis=-1) == flow_truth_y, tf.int32))

        #     # tf.summary.scalar('flow_error_if_predict_zeros', tf.reduce_mean(0.5 * tf.square(flow_truth)))
        #     tf.summary.scalar('flow_x_loss', flow_x_loss)
        #     tf.summary.scalar('flow_y_loss', flow_y_loss)
        #     tf.summary.scalar('flow_x_acc', flow_x_acc)
        #     tf.summary.scalar('flow_y_acc', flow_y_acc)
        #     # tf.summary.image('predicted_flow_x', tf.expand_dims(predicted_flow[..., 0], axis=-1), max_outputs=1)
        #     # tf.summary.image('predicted_flow_y', tf.expand_dims(predicted_flow[..., 1], axis=-1), max_outputs=1)

        self.train_writer = tf.summary.FileWriter(os.path.join(hparams['base_dir'], 'logs', hparams['experiment_name']), sess.graph)
        # TODO(vikgoel): when we don't need the teacher, we should ensure that we don't merge its summaries so that way
        #                we don't need to execute that part of the graph.
        merged_summaries = tf.summary.merge_all()

        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        def get_train_op(loss_op):
            params = find_trainable_variables("model")

            # Switch from GATE_NONE to GATE_GRAPH to enhance reproducibility.
            #grads = tf.gradients(loss, params)
            grads_and_params = trainer.compute_gradients(loss=loss_op, var_list=params, gate_gradients=tf.train.RMSPropOptimizer.GATE_GRAPH)
            grads = [x[0] for x in grads_and_params]
            params = [x[1] for x in grads_and_params]

            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))

            return trainer.apply_gradients(grads)

        _fast_train = get_train_op(loss)
        _teacher_train = get_train_op(loss + teacher_loss)

        params = find_trainable_variables("model")
        print('*' * 20)
        print('chosen trainable variables')
        for p in params:
            print(p.name)
        print('*' * 20)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.lr = lr

        write_counter = 0


        def train(obs, states, rewards, masks, actions, values):
            nonlocal write_counter

            if lr.n % hparams['target_model_update_frequency'] == 0 and hasattr(self, 'target_model'):
                print('COPYING WEIGHTS INTO TARGET MODEL')
                self.target_model.copy_weights()

            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            # Smooth approximation:
            #teacher_decay_c = hparams['teacher_decay_c']#9.9e-6 # 2.5e-5
            #teacher_c = 1.0 / (teacher_decay_c * lr.n + 1)
            #teacher_c = min(hparams['max_teacher_c'], teacher_c)

            if not hparams['use_extra_path']:
                lerp = float(lr.n) / 1e7
                lerp = min(lerp, 1)
                teacher_c = hparams['max_teacher_c'] * (1. - lerp)
            else:
                teacher_c = 1

            # Linear decay schedule
            # teacher_c = (hparams['teacher_cutoff_step'] - lr.n) / hparams['teacher_cutoff_step']
            # teacher_c = max(teacher_c, 0)

            # # Lower bound on the decay
            # teacher_c = (1 - hparams['teacher_loss_c']) * teacher_c + hparams['teacher_loss_c']

            _train = _fast_train if teacher_c == 0 else _teacher_train

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, TEACHER_C:teacher_c}
            # td_map[DROPOUT_STRENGTH] = get_dropout_strength(hparams, lr.n)

            if self.hparams['teacher_ckpt'] and self.hparams['do_joint_training']:
                td_map[train_teacher.mask_reg_c] = 1

            #if states is not None:
            #    td_map[train_model.S] = states
            #    td_map[train_model.M] = masks

            ops = [pg_loss, vf_loss, entropy, _train]

            # if hparams.get('no_train_a2c'):
            #     ops = ops[:-1]

            if 'attention' in hparams['policy']:
                ops.append(train_model.attention_weights_20)

            write_summaries = hparams.get('teacher_ckpt') or 'attention' in hparams['policy']

            if write_summaries:
                if write_counter % 10 != 0:
                    write_summaries = False
                write_counter += 1

            if write_summaries:
                ops.append(merged_summaries)

            sess_results = sess.run(ops, td_map)

            policy_loss = sess_results[0]
            value_loss = sess_results[1]
            policy_entropy = sess_results[2]

            if write_summaries:
                summary = sess_results[-1]
                self.train_writer.add_summary(summary, lr.n)

            if 'attention' in hparams['policy']:
                attention_output = sess_results[-2 if write_summaries else -1]
                publish_attention_weights(attention_output[:5,...])

            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load


        # Initialize all of the variables in a deterministic order so that each experiment is reproducible.
        global_vars = tf.global_variables()
        global_vars = sorted(global_vars, key=lambda x: x.name)
        for var in global_vars:
            tf.variables_initializer([var]).run(session=sess)
        #tf.global_variables_initializer().run(session=sess)

        if hparams.get('teacher_ckpt'):
            # Load in the teacher AFTER doing the init so we don't overwrite the weights.
            restore_teacher_from_checkpoint(sess, hparams['teacher_ckpt'])



class Runner(object):

    def __init__(self, hparams, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.hparams = hparams

        self.nenvs = nenv
        self.nbatch = nenv * nsteps

    def run(self):
        if hasattr(self.model, 'step_env_random'):
            step_env_random = self.model.step_env_random
            sample_normal_op = tf.truncated_normal(shape=[7, 7, 1])
            new_normal_placeholder = tf.placeholder(shape=[self.nenvs, 7, 7, 1], dtype=tf.float32)
            assign_normal_op = tf.assign(ref=step_env_random, value=new_normal_placeholder)


        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        dropout_strength_tup = (self.model.DROPOUT_STRENGTH, get_dropout_strength(self.hparams, self.model.lr.n + self.nbatch))

        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones, _dropout_strength=dropout_strength_tup)

            if hasattr(self.model, 'target_model'):
                values = self.model.target_model.value(self.obs, self.states, self.dones, _dropout_strength=dropout_strength_tup).tolist()

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0

                    if hasattr(self.model, 'step_env_random'):
                        [cur_rand, new_rand] = self.model.sess.run([step_env_random, sample_normal_op])
                        cur_rand[n] = new_rand
                        self.model.sess.run(assign_normal_op, feed_dict={new_normal_placeholder: cur_rand})

            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if hasattr(self.model, 'target_model'):
            last_values = self.model.target_model.value(self.obs, self.states, self.dones, _dropout_strength=dropout_strength_tup).tolist()
        else:
            last_values = self.model.value(self.obs, self.states, self.dones, _dropout_strength=dropout_strength_tup).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

def restore_from_checkpoint(sess, ckpt_path, first_layer_mode, hparams):
    var_names = [
        'model/c1_2_frame_input/w',
        'model/c1_2_frame_input/b',

        'model/c1/w',
        'model/c1/b',
        'model/c2/w',
        'model/c2/b',
        'model/c3/w',
        'model/c3/b',
        'model/fc1/w',
        'model/fc1/b',

        'model/fc2/w',
        'model/fc2/b',

        'model/conv4/w',
        'model/conv4/b',
        'model/conv5/w',
        'model/conv5/b',
        'model/conv6/w',
        'model/conv6/b',
        'model/masks/w',
        'model/masks/b',

        'model/flow/camera_translation/w',
        'model/flow/camera_translation/b',

        'model/object_translation/obj_t/w',
        'model/object_translation/obj_t/b',

        'model/upsample1/conv4/w',
        'model/upsample1/conv4/b',
        'model/upsample2/conv5/w',
        'model/upsample2/conv5/b',
        'model/upsample3/conv6/w',
        'model/upsample3/conv6/b',
        'model/upsample4/conv7/w',
        'model/upsample4/conv7/b',

        'model/object_masks/masks/w',
        'model/object_masks/masks/b',
    ]

    if hparams.get('pretrain_vf'):
        var_names += ['model/v/w', 'model/v/b']

    var_dict = {}
    for var_name in var_names:
        var_options = [v for v in tf.global_variables() if v.name == var_name+':0']
        if var_options:
            print('restored {}'.format(var_name))
            var_dict[var_name] = var_options[0]
        else:
            print('could not restore {}'.format(var_name))

    saver = tf.train.Saver(var_list=var_dict)
    do_restore(sess, saver, ckpt_path)

def log_scalars(writer, step):
    def log_scalar(name, value):
        logger.record_tabular(name, value)

        summary = tf.Summary(value=[tf.Summary.Value(tag=name,
                                                         simple_value=value)])
        writer.add_summary(summary, step)
    return log_scalar

def learn(
    policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5,
    ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5,
    alpha=0.99, gamma=0.99, log_interval=100, ckpt_path='', hparams=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    if hparams.get('atari_lr'):
        lr = hparams['atari_lr']

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule,
        hparams=hparams)
    runner = Runner(hparams, env, model, nsteps=nsteps, gamma=gamma)

    print('*'*20)
    print('all trainable variable names:')
    for var in tf.trainable_variables():
        print(var.name)
    print('*'*20)

    if ckpt_path:
        restore_from_checkpoint(model.sess, ckpt_path, hparams['first_layer_mode'], hparams)

    if hparams.get('student_ckpt'):
        saver = tf.train.Saver()
        do_restore(model.sess, saver, hparams['student_ckpt'])

    # Setup for saving checkpoints.
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0, max_to_keep=3)
    ckpt_dir = os.path.join(hparams['base_dir'], 'ckpts', hparams['experiment_name'])
    try: os.makedirs(ckpt_dir)
    except: pass

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)

        if update % 10 == 0:
            log_scalar = log_scalars(model.train_writer, update)
            log_scalar('nupdates', update)
            log_scalar('total_timesteps', update*nbatch)
            log_scalar('fps', fps)
            log_scalar('policy_entropy', float(policy_entropy))

            ev = explained_variance(values, rewards)
            log_scalar('explained_variance', float(ev))

            log_scalar('policy_loss', float(policy_loss))
            log_scalar('value_loss', float(value_loss))
            log_scalar('mean_reward', float(np.mean(rewards)))

            logger.dump_tabular()

        if (update-1) % hparams['save_interval'] == 0:
            saver.save(model.sess, os.path.join(ckpt_dir, 'my-model'), global_step=update)

        if (update-1) % 25000 == 0:
            try:
                experiment_name = os.path.basename(hparams['base_dir'])
                instance_name = hparams['experiment_name']

                for seperator in ['ckpts/', 'logs/', 'videos/']:
                    if 'video' in seperator:
                        src_instance_name = instance_name + '_rgb'
                    else:
                        src_instance_name = instance_name

                    src_dir = os.path.join(hparams['base_dir'], seperator+src_instance_name)
                    dst_dir = 's3://irl-experiments/{}/{}'.format(experiment_name, seperator+instance_name)

                    command = 'aws s3 sync {} {}'.format(src_dir, dst_dir)
                    print(command)
                    os.system(command)
            except:
                pass

    env.close()
