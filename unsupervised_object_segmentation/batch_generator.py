"""Loads in the object segmentation dataset and yields training batches from it."""

import os
from PIL import Image
import glob
import random
import numpy as np
from make_dataset import make_dataset
from a2cutils import discount_with_dones

class BatchGenerator(object):

    def __init__(self, hparams):
        self.hparams = hparams

        batch_size = self.hparams['batch_size']
        img_h = self.hparams['img_h']
        img_w = self.hparams['img_w']
        self.num_frames = self.hparams['num_input_frames']

        if self.hparams['do_frame_prediction']:
            self.num_frames += self.hparams['num_prediction_frames']

        self.frames = np.zeros([batch_size, img_h, img_w, self.num_frames])
        self.actions = np.zeros([batch_size, self.num_frames, self.hparams['num_actions']])

        # TODO(vikgoel): read the entire data path out of the hparams file.
        data_path = os.path.join(hparams['data_dir'], hparams['env_id'], 'sfmnet/episodes/*')
        print('data path is {}'.format(data_path))

        if not os.path.exists(data_path[:-1]):
            print('dataset not already loaded, creating dataset...')
            make_dataset(hparams['env_id'])

        # Each episode is stored in its own folder.
        episode_paths = sorted(glob.glob(data_path))
        print('found {} episode paths'.format(len(episode_paths)))

        # Load in each episode in sorted order by frame.
        # Currently the format for each filename is frameindex_action.png.
        # So we extract the frameindex and sort by this.
        self.episodes = {}
        for episode_path in episode_paths:
            self.episodes[episode_path] = sorted(
                glob.glob(os.path.join(episode_path, '*.png')),
                key=lambda x: int(os.path.basename(x).split('_')[0]),
            )

        def calculate_state_values(frame_paths):
            rewards = []

            for frame_path in frame_paths:
                frame_name, _ = os.path.splitext(os.path.basename(frame_path))
                reward = float(frame_name.split('_')[2])
                rewards.append(reward)

            dones = [False] * len(rewards)
            dones[-1] = True
            gamma = 0.99

            values = discount_with_dones(rewards, dones, gamma)
            assert len(values) == len(frame_paths)

            return list(zip(frame_paths, values))

        all_episodes = sorted(self.episodes.keys())

        for k, v in self.episodes.items():
            self.episodes[k] = calculate_state_values(v)

        if hparams.get('sfmnet_preload_images'):
            # Preload all the images into memory for better performance.
            print('preloading images...')
            for i, (episode, frames) in enumerate(self.episodes.items()):
                if i % 10 == 0:
                    print('{} / {}'.format(i, len(self.episodes)))
                self.episodes[episode] = [(self._open_image(frame[0]), frame[1]) for frame in frames]
            # Replace open image with the identity function since all the images are already loaded.
            self._open_image = lambda x: x

        if hparams.get('sfmnet_small_dataset'):
            train_episodes = all_episodes[20:30]
            val_episodes = [all_episodes[0]]
        else:
            # TODO(vikgoel): more rigorous way of doing train/val/test split.
            train_val_split = int(.2 * len(all_episodes))
            train_episodes = all_episodes[train_val_split:]
            val_episodes = all_episodes[:train_val_split]

        # This fix was needed for Elevator action where we were only getting two episodes
        if len(val_episodes) == 0:
            print("Setting the val set to be the train set since we have fewer than 5 episodes")
            val_episodes = train_episodes

        self.train_batch_generator = self._make_generator(train_episodes)
        self.val_batch_generator = self._make_generator(val_episodes)

    def get_batch(self, data_type):
        if data_type == 'train':
            return next(self.train_batch_generator)
        elif data_type == 'val':
            return next(self.val_batch_generator)
        else:
            raise NotImplementedError

    def _make_generator(self, episode_keys):
        global dedup_action_map

        episodes = {k : self.episodes[k] for k in episode_keys}

        # Want to sample episodes proportional to the length of each episode.
        valid_keys = [k for k in episodes.keys() if len(episodes[k]) >= 2]
        probs = np.array([len(episodes[k]) for k in valid_keys], dtype=np.float32)
        probs /= np.sum(probs)

        state_values = np.zeros([self.hparams['batch_size']])

        while True:
            self.actions.fill(0)
            state_values.fill(0)

            # Sample consecutive frames to compose the batch.
            for i in range(self.hparams['batch_size']):
                key = np.random.choice(valid_keys, p=probs)
                frames = episodes[key]
                idx = random.randint(0, len(frames) - self.num_frames)

                for j in range(self.num_frames):
                    frame_path, value = frames[idx + j]
                    self.frames[i,:,:,j:j+1] = self._open_image(frame_path)

                    state_values[i] = value

            yield self.frames, self.actions, state_values


    def _open_image(self, path):
        return np.expand_dims(np.array(Image.open(path)), axis=-1)
