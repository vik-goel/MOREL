"""
Helpers for scripts like run_atari.py.
"""

import gym
import os
import imageio

import numpy as np
import cv2

from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

attention_mask_subscribers = []
my_subproc_vec_env = None

def publish_attention_weights(attention_weights):
    print('publish_attention_weights')

    global my_subproc_vec_env
    assert my_subproc_vec_env != None
    my_subproc_vec_env.send_attention_weights(attention_weights)

def on_attention_weights(attention_weights):
    print('on_attention_weights')

    global attention_mask_subscribers

    for subscriber in attention_mask_subscribers:
        subscriber.on_attention_weights(attention_weights)


class VideoLogMonitor(gym.Wrapper):
    episode_save_rate = 20
    num_instances = 0

    @classmethod
    def class_name(cls):
        cls.num_instances += 1
        return cls.__name__ + str(cls.num_instances)

    def __init__(self, env, video_dir, write_attention_video=False, hparams=None, nsteps=5):
        super().__init__(env)

        imageio.plugins.ffmpeg.download()

        self.episode_id = 0
        self.env_semantics_autoreset = env.metadata.get('semantics.autoreset')

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        self.video_dir = video_dir
        self.writer = None

        self.history = []
        self.write_attention_video = write_attention_video
        self.attention_writer = None

        self.attention_scale_const = 8
        if hparams != None and 'attention_video_recording_scale_const' in hparams:
            self.attention_scale_const = hparams['attention_video_recording_scale_const']

        if self.write_attention_video:
            global attention_mask_subscribers
            attention_mask_subscribers.append(self)

        self.nsteps = nsteps


    def _start_new_episode(self):
        if self.writer:
            self.writer.close()

        if self.attention_writer:
            self.attention_writer.close()

        if self.episode_id % self.episode_save_rate == 0:
            self.writer = imageio.get_writer(
                os.path.join(self.video_dir, 'video_episode_{}.mp4'.format(self.episode_id)),
                fps=15,
            )
            if self.write_attention_video:
                self.attention_writer = imageio.get_writer(
                    os.path.join(self.video_dir, 'video_episode_{}_attention.mp4'.format(self.episode_id)),
                    fps=15,
                )
        else:
            self.writer = None
            self.attention_writer = None

        self.episode_id += 1

    def _capture_frame(self):
        frame = self.env.render(mode='rgb_array')

        if self.writer is not None:
            self.writer.append_data(frame)

        self.history.append(frame)
        # NOTE: This should be larger than the number of frames we process before doing a weight update.
        self.history = self.history[-self.nsteps * 2:]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done and self.env_semantics_autoreset:
            self._start_new_episode()
        self._capture_frame()
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._start_new_episode()
        self._capture_frame()
        return observation

    def on_attention_weights(self, attention_weights):
        #print('video on_attention_weights({})'.format(len(self.history)))

        # [5, 7, 7, 1] vector as input

        if self.attention_writer is not None and self.history:
            #print('here1')

            # Use the history and attention weights to create frames.
            attention_weights = np.squeeze(attention_weights, axis=-1)

            num_attention_frames = attention_weights.shape[0]
            attention_frames = []

            #print(attention_weights.shape)

            for i in range(num_attention_frames):
                img_size = tuple(reversed(self.history[0].shape[:2]))
                frame = attention_weights[i, ...]

                # attn_h = frame.shape[1]
                # attn_w = frame.shape[2]

                # frame = np.reshape(frame, [attn_h*attn_w])
                # frame = np.log(frame)
                # frame = frame * 10
                # frame = np.exp(frame)
                # frame /= np.sum(frame)
                # frame = np.reshape(frame, [attn_h, attn_w])

                upsampled = cv2.resize(frame, img_size, interpolation=cv2.INTER_LANCZOS4)
                attention_frames.append(upsampled)

                #print('here1.5')

            #print('here2')

            FRAME_SKIP = 1
            ALPHA = 0.75

            write_count = 0

            for i, attention_frame in enumerate(attention_frames):
                frame_start = -(num_attention_frames * FRAME_SKIP) + i

                attention_frame = np.expand_dims(attention_frame, axis=-1)

                attention_frame = np.maximum(np.minimum(attention_frame, 1.), 0.)

                #print('here3.5')

                for j in range(FRAME_SKIP):
                    # TODO(vikgoel): indexing self.history was giving an index out of bounds exception,
                    #                try except was not the best solution, should investigate this further
                    try:
                        history_index = frame_start + j


                        attention_image = np.minimum(attention_frame * self.attention_scale_const, 1.0)

                        frame = ALPHA * attention_image * 255 + (1.-ALPHA)*self.history[history_index] 

                        frame = np.asarray(frame, dtype=np.uint8)
                        self.attention_writer.append_data(frame)
                        write_count += 1
                    except Exception as e:
                        print(e)
                        pass

            #print("WROTE FRAMES: {}".format(write_count))

            self.history = []




def make_atari_env(env_id, num_env, seed, hparams=None, wrapper_kwargs=None, start_index=0, nsteps=5, **kwargs):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

            if rank == start_index and 'video_log_dir' in kwargs:
                env = VideoLogMonitor(env, kwargs['video_log_dir'] + '_rgb', write_attention_video=kwargs['write_attention_video'], hparams=hparams, nsteps=nsteps)

            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)

    env_fns = [make_env(i + start_index) for i in range(num_env)]

    global my_subproc_vec_env
    assert my_subproc_vec_env == None
    my_subproc_vec_env = SubprocVecEnv(env_fns)

    return my_subproc_vec_env

def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser

def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default="Reacher-v1")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser
