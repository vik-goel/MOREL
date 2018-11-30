"""Train the unsupervised video object segmentation network."""

import os
# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import sys
import json

with open(sys.argv[1], 'r') as f:
    hparams = json.load(f)

if len(sys.argv) > 2:
    gpu_num = str(sys.argv[2])
    assert int(gpu_num) >= -1 and int(gpu_num) <= 8
    print("GPU NUM: {}".format(gpu_num))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
elif 'gpu_num' in hparams:
    gpu_num = str(hparams['gpu_num'])
    assert int(gpu_num) >= -1 and int(gpu_num) <= 8
    print("GPU NUM: {}".format(gpu_num))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

import shutil
import glob
import numpy as np
import tensorflow as tf
import copy

_opencv_path = '/home/v5goel/test_env/env/lib/python3.5/site-packages/'
if os.path.exists(_opencv_path):
    sys.path.insert(0, _opencv_path)

from object_segmentation_network import ObjectSegmentationNetwork, ObjectSegmentationNetworkPredict
from batch_generator import BatchGenerator



import gym
hparams['num_actions'] = gym.make(hparams['env_id']).action_space.n

if hparams['do_frame_prediction']:
    motion_model = ObjectSegmentationNetworkPredict(hparams=copy.deepcopy(hparams))
else:
    motion_model = ObjectSegmentationNetwork(hparams=copy.deepcopy(hparams))

batch_generator = BatchGenerator(hparams=copy.deepcopy(hparams))

# Setup for saving checkpoints.
saver = tf.train.Saver(max_to_keep=3)
ckpt_dir = os.path.join(hparams['base_dir'], 'ckpts', hparams['experiment_name'])
try: os.makedirs(ckpt_dir)
except: pass

# Copy the hparams into the ckpt dir.
try: shutil.copy(sys.argv[1], ckpt_dir)
except: pass

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=hparams.get('gpu_fraction', 0.25))

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Setup for saving logs.
    merged_summaries = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(
        os.path.join(hparams['base_dir'], 'logs', 'train_' + hparams['experiment_name']), sess.graph)

    val_writer = tf.summary.FileWriter(
        os.path.join(hparams['base_dir'], 'logs', 'val_' + hparams['experiment_name']), sess.graph)

    sess.run(tf.global_variables_initializer())

    if hparams.get('restore_from_ckpt_path'):
        saver = tf.train.Saver(max_to_keep=3)
        restore_dir = hparams['restore_from_ckpt_path']
        ckpt_path = tf.train.latest_checkpoint(restore_dir)
        assert ckpt_path is not None
        saver.restore(sess, ckpt_path)

    training_steps = hparams['total_timesteps']
    print('training_steps is {}'.format(training_steps))

    print('*'*20)
    print('trainable variable names:')
    for var in tf.trainable_variables():
        print(var.name)
    print('*'*20)

    # Train loop.
    for step in range(training_steps):
        if step > 2.5e5 + 10:
            print('DONE TRAINING')
            break

        try:
            def get_feed_dict(train_or_val):
                frames, actions, state_values = batch_generator.get_batch(train_or_val)
                feed_dict = {}
                feed_dict[motion_model.frames_placeholder] = frames
                feed_dict[motion_model.learning_rate] = motion_model.next_learning_rate()

                if hparams.get('pretrain_vf'):
                    feed_dict[motion_model.value_placeholder] = state_values

                mask_lerp = min(1, float(step) / 1e5)
                mask_lerp = max(0, mask_lerp)

                assert mask_lerp >= 0
                assert mask_lerp <= 1
                feed_dict[motion_model.mask_reg_c] = hparams['mask_reg_loss_c'] * mask_lerp

                if hparams['do_frame_prediction']:
                    feed_dict[motion_model.actions] = actions

                return feed_dict

            # Train on batch.
            ops = [motion_model.total_loss, motion_model.train_op]

            save_summary = step % 10 == 0
            if save_summary:
                ops.append(merged_summaries)

            sess_result = sess.run(ops, feed_dict=get_feed_dict('train'))
            train_loss = sess_result[0]

            if save_summary:
                summary = sess_result[-1]
                train_writer.add_summary(summary, step)

            # Periodically run on validation data.
            if step % hparams['val_interval'] == 0:
                val_loss, summary = sess.run([motion_model.total_loss, merged_summaries], feed_dict=get_feed_dict('val'))
                val_writer.add_summary(summary, step)
                print("step: {}, train loss: {}, val loss: {}".format(step, train_loss, val_loss))

            # Periodically checkpoint the model.
            if step % hparams['save_interval'] == 0:
                saver.save(sess, os.path.join(ckpt_dir, 'my-model'), global_step=step)

            if step % 50000 == 0:
                try:
                    experiment_name = os.path.basename(hparams['base_dir'])
                    instance_name = hparams['experiment_name']

                    for seperator in ['ckpts/', 'logs/train_', 'logs/val_']:
                        src_dir = os.path.join(hparams['base_dir'], seperator+instance_name)
                        dst_dir = 's3://irl-experiments/{}/{}'.format(experiment_name, seperator+instance_name)

                        command = 'aws s3 sync {} {}'.format(src_dir, dst_dir)
                        print(command)
                        os.system(command)
                except:
                    pass

        except Exception as e:
            print('exception in train loop: {}'.format(e))
            raise
