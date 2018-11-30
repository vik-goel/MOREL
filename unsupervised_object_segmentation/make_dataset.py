"""Creates the object segmentation dataset."""

import numpy as np
import os
import sys
from PIL import Image
import sys

for path in ['src/', 'interpretable-rl/src']:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

import gym

if not '..' in sys.path:
    sys.path.insert(0, '..')

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common import set_global_seeds

def make_dataset(env_name):
    # Hyper-parameters
    total_frames_to_generate = 100000
    #env_id = ['PongNoFrameskip-v4', 'SeaquestNoFrameskip-v4'][1]
    env_id = env_name
    save_path = './data/{}/sfmnet/episodes'.format(env_id)
    seed = 0

    # Track how many frames we have created.
    total_frames_generated = 0
    episode_index = 0

    # Create and set-up the environment.
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env.seed(seed)
    set_global_seeds(seed)

    # Generate frames.
    while total_frames_generated < total_frames_to_generate:
        print("Starting episode {}".format(episode_index))

        obs = env.reset()
        frame_index = 0
        done = False

        while not done and total_frames_generated < total_frames_to_generate:
            # Take a random action.
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            # Create a directory to save frames to for this episode.
            episode_save_path = os.path.join(save_path, str(episode_index))
            if not os.path.exists(episode_save_path):
                os.makedirs(episode_save_path)

            # Save the frame
            img = Image.fromarray(np.squeeze(obs), mode='L')
            img.save(os.path.join(episode_save_path, '{}_{}_{}.png'.format(frame_index, action, reward)))
            frame_index += 1
            total_frames_generated += 1

        # Start a new episode.
        episode_index += 1

if __name__ == '__main__':
    make_dataset(sys.argv[1] + 'NoFrameskip-v4')
