#!/usr/bin/env python3
import sys
import json

import os
# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

for path in ['src/', 'interpretable-rl/src', 'src/unsupervised_object_segmentation', 'interpretable-rl/src/unsupervised_object_segmentation']:
    if os.path.exists(path):
        sys.path.insert(0, path)

_opencv_path = '/home/v5goel/test_env/env/lib/python3.5/site-packages/'
if os.path.exists(_opencv_path):
    sys.path.insert(0, _opencv_path)

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, CnnAttentionPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, ckpt_path, hparams):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'cnn_attention':
        policy_fn = CnnAttentionPolicy

    video_log_dir = os.path.join(hparams['base_dir'], 'videos', hparams['experiment_name'])
    env = VecFrameStack(make_atari_env(env_id, num_env, seed, video_log_dir=video_log_dir, write_attention_video='attention' in policy, hparams=hparams), 4)

    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, ckpt_path=ckpt_path, hparams=hparams)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
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

    print('experiment_params: {}'.format(hparams))
    print('chosen env: {}'.format(hparams['env_id']))

    seed = 0
    if hparams.get('atari_seed'): seed = hparams['atari_seed']

    logger.configure(dir=log_path)
    train(
        env_id=hparams['env_id'],
        num_timesteps=hparams['total_timesteps'],
        seed=seed,
        policy=hparams['policy'],
        lrschedule=args.lrschedule,
        num_env=hparams['num_env'],
        ckpt_path=hparams['restore_from_ckpt_path'],
        hparams=hparams,
    )

if __name__ == '__main__':
    main()
