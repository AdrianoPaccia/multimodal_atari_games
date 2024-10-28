import os
import random
import numpy as np
import torch
from gymnasium import spaces
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import matplotlib.pyplot as plt
import matplotlib
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
class WalkerEnv:

    def __init__(
            self,
            noise_generators: dict ={},
            max_episode_steps: int = 300,
            noise_frequency: float = 0.0,
            n_noisy_obs: int = 1
    ):
        self.obs_modes = ('state', 'rgb')

        if not set(tuple(noise_generators.keys())) <= set(self.obs_modes):
            raise ValueError('noise_generators keys are not a subset of the obs_modes')
        else:
            self.noise_generators = noise_generators
        self.max_episode_steps = max_episode_steps
        self.noise_frequency = noise_frequency

        self.device = torch.device('cpu')
        if n_noisy_obs > len(self.noise_generators) or n_noisy_obs < 0:
            raise ValueError('n_noisy_obs must not be greater than the number of modes')
        else:
            self.n_noisy_obs = n_noisy_obs

        #built env
        self.env = PixelObservationWrapper(
            env=gym.make('Walker2d-v4', render_mode='rgb_array', max_episode_steps=max_episode_steps),
            pixels_only=False,
            pixel_keys=("rgb",),
        )

        self.single_state_shape = self.env.observation_space['state'].shape
        self.state_space = self.env.observation_space['state']

        self.single_observation_space_mm = spaces.Tuple([
            self.state_space,  # state
            spaces.Box(low=0, high=255, shape=(100,100,3)),  # image
        ])

        self.observation_space_mm = spaces.Tuple([
            spaces.Box(low=-10., high=10., shape=(1,)+self.single_state_shape),  # state
            spaces.Box(low=0, high=255, shape=(1,100,100,3)),  # image
        ])
        self.action_space = self.env.action_space
        self.single_action_space = self.env.action_space


    def step(self, a):
        """Method for step in the parent environment"""
        timestep = self.env.step(a)
        #truncated = self.env._step_count > self.max_episode_steps
        self.ep_reward += timestep.reward
        return timestep.observation, timestep.reward, timestep.last(), truncated, {}

    def step_mm(self, a):
        """Method for step in the multimodal environment"""
        if torch.is_tensor(a):
            a = a.numpy().reshape(-1)

        self.observation, reward, done, truncated, info = self.env.step(a)

        self.ep_reward += reward
        self.ep_step +=1

        #if self.env._step_count >= self.max_episode_steps:
        #    truncated = True

        #assemble the obs
        obs = dict(
            state=self.observation['state'].copy(),
            rgb=self.observation['rgb'].copy(),
        )

        # inject noise
        if random.random() < self.noise_frequency:
            for m in random.sample(list(self.noise_generators.keys()), self.n_noisy_obs):
                obs[m] = self.noise_generators[m].get_observation(obs[m])
        obs['state'] = torch.from_numpy(obs['state']).unsqueeze(0)
        img = Image.fromarray(obs['rgb'])
        obs['rgb'] = torch.from_numpy(np.asarray(img.resize((100,100)))).unsqueeze(0)

        reward = torch.tensor([reward]).unsqueeze(0)
        done = torch.tensor([done]).unsqueeze(0)
        truncated = torch.tensor([truncated]).unsqueeze(0)

        info = {
            'elapsed_steps': torch.tensor([self.ep_step]),
            'episode': {'r': torch.tensor([self.ep_reward])}
        }

        if done or truncated:
            info['final_info'] = {
                'elapsed_steps': torch.tensor([self.ep_step]),
                'episode': {
                    'r': torch.tensor([self.ep_reward]),
                    '_r': torch.tensor([True])
                },
            }
            info['_final_info'] = torch.tensor([True])
            info['final_observation'] = obs

        return obs, reward, done, truncated, info

    def render(self):
        matplotlib.use('TkAgg')
        ax = plt.gca()
        ax.clear()
        img = self.observation['rgb']#self.env._env.physics.render()
        ax.imshow(img)
        plt.draw()
        plt.pause(0.01)
        return img

    def reset(self):
        """Method for resetting the parent environment"""
        self.ep_reward = 0.
        self.ep_step = 0
        return self.env.reset()

    def reset_mm(self, seed=0, num_initial_steps=1):
        """Method for resetting the multimodal environment"""
        self.reset()

        if type(num_initial_steps) is list or type(num_initial_steps) is tuple:
            assert len(num_initial_steps) == 2
            low = num_initial_steps[0]
            high = num_initial_steps[1]

            num_initial_steps = np.random.randint(low, high)
        elif type(num_initial_steps) is int:
            assert num_initial_steps >= 1
        else:
            raise 'Unsupported type for num_initial_steps. Either list/tuple or int'

        for _ in range(num_initial_steps):
            obs, _, _, _, info = self.step_mm(self.action_space.sample())

        return obs, info

    def close(self):
        """Method for closing the environment"""
        self.env.close()

    def get_state(self):
        """Method for getting the current state"""
        return torch.from_numpy(self.observation['state']).unsqueeze(0)

    def show_description(self):
        print(f"{'-'*20} DESCRIPTION {'-'*20}")
        print(f" - game: {self.game}")
        print(f" - task: {self.task}")
        print(f" - modes:")
        for i, m in enumerate(self.obs_modes):
            print(f"   + '{m}' of shape {self.single_observation_space_mm[i].shape}")
        print(f" - state_keys: {self.state_keys}")
        print(f" - noisy obs: {self.n_noisy_obs} of {list(self.noise_generators.keys())}")
        print(f"{'-'*53}")