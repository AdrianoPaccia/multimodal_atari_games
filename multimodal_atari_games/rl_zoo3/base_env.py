import os
import random
import numpy as np
import torch
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib
import yaml

from rl_zoo3 import create_test_env, ALGOS, get_saved_hyperparams
import multimodal_atari_games.multimodal_atari_games.rl_zoo3.import_envs
import cv2
os.environ["MUJOCO_GL"] = "egl"

with open(f'{os.path.dirname(os.path.realpath(__file__))}/configurations.yaml') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

class BaseRLzoo3Env:

    def __init__(
            self,
            env_id='FetchPush-v1',
            args=dict(hyperparams={}, kwargs={}, algo='tqc'),
            noise_generators: dict = {},
            max_episode_steps: int = 300,
            noise_frequency: float = 0.0,
            n_noisy_obs: int = 1,
            **kwargs
    ):
        self.env_id = env_id
        self.obs_modes = tuple(args['modes'])
        algo = args['algo']
        env_args = args['env']
        self.num_envs = kwargs['num_envs']

        if not set(tuple(noise_generators.keys())) <= set(self.obs_modes):
            raise ValueError('noise_generators keys are not a subset of the obs_modes')
        else:
            self.noise_generators = noise_generators
        self.max_episode_steps = max_episode_steps
        self.noise_frequency = noise_frequency

        self.device = torch.device('cpu')

        if len(self.noise_generators) == 0:
            n_noisy_obs = 0

        if n_noisy_obs > len(self.noise_generators) or n_noisy_obs < 0:
            raise ValueError('n_noisy_obs must not be greater than the number of modes')
        else:
            self.n_noisy_obs = n_noisy_obs

        #build ENVIRONMENT
        env = create_test_env(
            env_id,
            n_envs=kwargs['num_envs'] if 'num_envs' in kwargs else None,
            stats_path=None,
            seed=kwargs['seed'] if 'seed' in kwargs else None,
            log_dir=None,
            should_render=False,
            hyperparams=env_args["hyperparams"],
            env_kwargs=env_args["kwargs"],
        )
        self.env = ALGOS[algo]._wrap_env(env, False)
        self.dense_reward = env_args["dense_reward"]

        self.state_keys = kwargs['num_envs'] if 'state_keys' in kwargs else list(self.env.observation_space.keys())

        #set the spaces
        init_obs = dict(
            state=self.get_state(self.env.reset()),
            rgb=self.env.render()
        )

        _obs_space = self.env.observation_space
        self.single_state_shape = np.sum([np.array(_obs_space[k].shape) for k in _obs_space.keys()]),
        self.single_state_space = spaces.Box(
            low=np.concatenate([np.array(_obs_space[k].low) for k in _obs_space.keys()]),
            high=np.concatenate([np.array(_obs_space[k].high) for k in _obs_space.keys()]),
            shape=self.single_state_shape
        )
        self.state_space = spaces.Box(
            low=np.stack([self.single_state_space.low]*self.num_envs),
            high=np.stack([self.single_state_space.high]*self.num_envs),
            shape=(self.num_envs,) + self.single_state_shape
        )

        self.single_observation_space_mm = spaces.Tuple([
            self.single_state_space,  # state
            spaces.Box(low=0, high=255, shape=init_obs['rgb'].shape),  # image
        ])

        self.observation_space_mm = spaces.Tuple([
            self.state_space,  # state
            spaces.Box(low=0, high=255, shape=(self.num_envs,) + init_obs['rgb'].shape),  # depth
        ])

        self.single_action_space = self.env.action_space
        self.action_space = spaces.Box(
            low=np.stack([self.env.action_space.low]*self.num_envs),
            high=np.stack([self.env.action_space.high]*self.num_envs),
            shape=(self.num_envs,) + self.single_action_space.shape
        )


    def flatten_obs(self, obs):
        return np.concatenate([obs[k] for k in obs.keys()], axis=-1).reshape(self.num_envs, -1)


    def step(self, a):
        """Method for step in the parent environment"""
        obs, reward, done, info = self.env.step(a)
        self.ep_reward += reward
        truncated = True if self.ep_step >= self.max_episode_steps else False
        return obs, reward, done, truncated, info

    def step_mm(self, a):
        """Method for step in the multimodal environment"""

        if torch.is_tensor(a):
            a = a.numpy().reshape(1, -1)

        observation, reward, terminated, truncated, info = self.step(a)
        #reward = self.compute_dense_reward() if self.dense_reward else reward
        done = terminated or truncated

        #assemble the obs
        img = self.env.render()
        obs = dict(
            state=self.flatten_obs(observation),
            rgb=img.reshape(self.num_envs, *img.shape),
        )

        # inject noise
        if random.random() < self.noise_frequency:
            for m in random.sample(list(self.noise_generators.keys()), self.n_noisy_obs):
                obs[m] = self.noise_generators[m].get_observation(obs[m])

        #obs = {m: torch.from_numpy(o).unsqueeze(0) for m, o in obs.items()}
        #reward = torch.tensor([reward]).unsqueeze(0)
        #done = torch.tensor([done]).unsqueeze(0)
        #truncated = torch.tensor([truncated]).unsqueeze(0)

        info = {
            'is_success': [inf['is_success'] for inf in info],
            'elapsed_steps': [self.ep_step],
            'episode': {'r': self.ep_reward}
        }

        return obs, reward, done, truncated, info

    def render(self):
        matplotlib.use('TkAgg')
        ax = plt.gca()
        ax.clear()
        img = self.env.render()
        ax.imshow(img)
        plt.draw()
        plt.pause(0.01)
        return img

    def reset(self, seed=None):
        """Method for resetting the parent environment"""
        self.ep_step, self.ep_reward = 0, 0.
        self.env.seed = seed
        return self.env.reset()

    def reset_mm(self, seed=None):
        """Method for resetting the multimodal environment"""
        observation = self.reset(seed=seed)
        info = dict(
            elapsed_steps=torch.tensor([self.ep_step]),
            episode={'r': torch.tensor([self.ep_reward])}
        )

        img = self.env.render()
        obs = dict(
            state=self.flatten_obs(observation),
            rgb=img.reshape(self.num_envs, *img.shape),
        )
        return obs, info

    def close(self):
        """Method for closing the environment"""
        self.env.close()

    def get_state(self, obs=None):
        """Method for getting the current state""" #DONE
        obs = self.env.buf_obs() if obs is None else obs
        return torch.from_numpy(self.flatten_obs(obs))

    def show_description(self):
        print(f"{'-'*20} DESCRIPTION {'-'*20}")
        print(f" - game: {self.env_id}")
        print(f" - num. environments: {self.num_envs}")
        print(f" - modes:")
        for i, m in enumerate(self.obs_modes):
            print(f"   + '{m}' of shape {self.single_observation_space_mm[i].shape}")
        print(f" - state keys: {self.state_keys}")
        print(f" - noisy obs: {self.n_noisy_obs} of {list(self.noise_generators.keys())}")
        print(f"{'-'*53}")

    def compute_dense_reward(self):
        return -1
