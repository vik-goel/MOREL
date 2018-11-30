#!/usr/bin/env python3

import os
# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import sys

for path in ['src/', 'interpretable-rl/src', 'src/unsupervised_object_segmentation', 'interpretable-rl/src/unsupervised_object_segmentation']:
    if os.path.exists(path):
        sys.path.insert(0, path)

_opencv_path = '/home/v5goel/test_env/env/lib/python3.5/site-packages/'
if os.path.exists(_opencv_path):
    sys.path.insert(0, _opencv_path)

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2


from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, CnnAttentionPolicy
import multiprocessing
import tensorflow as tf
import json


def train(env_id, num_timesteps, seed, policy, hparams):

    ncpu = multiprocessing.cpu_count()
    #if sys.platform == 'darwin': ncpu //= 2
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=hparams['gpu_fraction'])
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu,
                            gpu_options=gpu_options)
    config.gpu_options.allow_growth = False #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    video_log_dir = os.path.join(hparams['base_dir'], 'videos', hparams['experiment_name'])
    env = VecFrameStack(make_atari_env(env_id, 8, seed, video_log_dir=video_log_dir, write_attention_video='attention' in policy, nsteps=128), 4)
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'cnn_attention': CnnAttentionPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        hparams=hparams)

def main():
    parser = atari_arg_parser()
    parser.add_argument('--hparams_path', help='Load json hparams from this file', type=str, default='')
    parser.add_argument('--gpu_num', help='cuda gpu #', type=str, default='')

    args = parser.parse_args()

    with open(args.hparams_path, 'r') as f:
        hparams = json.load(f)

    if args.gpu_num:
        assert(int(args.gpu_num) >= -1 and int(args.gpu_num) <= 8)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    elif 'gpu_num' in hparams:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.get('gpu_num'))

    log_path = os.path.join(hparams['base_dir'], 'logs', hparams['experiment_name'])
    logger.configure(dir=log_path)

    print('experiment_params: {}'.format(hparams))
    print('chosen env: {}'.format(hparams['env_id']))

    seed = 0
    if hparams.get('atari_seed'): seed = hparams['atari_seed']

    train(hparams['env_id'], num_timesteps=args.num_timesteps, seed=seed,
        policy=hparams['policy'], hparams=hparams)

if __name__ == '__main__':
    main()
