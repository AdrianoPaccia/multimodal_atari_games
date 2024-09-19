import os
import random
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import numpy as np
import torch
from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise
from gym import spaces
from PIL import Image
import copy

class CheetahImageConfiguration(HalfCheetahEnv):

    def __init__(
            self,
            render_mode='rgb_array',
            #ram_noise_generator=RamNoise([],0.0, game='pong'),
            image_noise_generator=ImageNoise(noise_types=[], game='cheetah'),
            max_episode_steps=300,
            noise_frequency=0.0
    ):
        super().__init__(render_mode=render_mode)
        self.render_mode = render_mode
        #self.ram_noise_generator=ram_noise_generator
        self.image_noise_generator=image_noise_generator
        self.max_episode_steps = max_episode_steps
        self.noise_frequency = noise_frequency
        self.ep_step = 0
        self.ep_reward = 0.
        self.device = torch.device('cpu')
        self.obs_modes = ['state','rgb']
        state_shape = (17,)
        img_shape = (100, 100, 3)
        self.single_state_shape = self.observation_space.shape

        self.observation_space_mm = spaces.Tuple([
            spaces.Box(low=-8, high=8, shape=state_shape),  # state
            spaces.Box(low=0, high=255, shape=img_shape),  # image
        ])

        self.single_observation_space_mm = self.observation_space_mm
        self.single_action_space = copy.deepcopy(self.action_space)

    def step(self, a):
        observation, reward, done, truncated, info = super().step(a)
        truncated = self.ep_step > self.max_episode_steps
        self.ep_step += 1
        self.ep_reward += reward
        return observation, reward, done, truncated, info

    def step_mm(self, a):

        if torch.is_tensor(a):
            a = a.numpy().reshape(-1)

        ram_observation, reward, done, truncated, info = self.step(a)

        self.ep_step += 1

        if self.ep_step >= self.max_episode_steps:
            truncated = True

        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)
        truncated = torch.tensor(truncated).unsqueeze(0)

        info = {
            'elapsed_steps': torch.tensor([self.ep_step]),
            'episode': {'r': torch.tensor([self.ep_reward])}
        }

        if self.render_mode != 'rgb_array':
            obs = dict(state=torch.tensor(ram_observation).unsqueeze(0))

        else:
            img_observation = super().render()[180:480, 100:400]
            if random.random() < self.noise_frequency:
                img_observation = self.image_noise_generator.get_observation(img_observation)
                # if bool(random.getrandbits(1)):
                #    img_observation = self.image_noise_generator.get_observation(img_observation)
                # else:
                #    ram_observation = self.ram_noise_generator.get_observation(ram_observation)

            rgb = Image.fromarray(img_observation)
            img_observation = np.array(rgb.resize((100, 100)))

            obs = dict(
                state=torch.tensor(ram_observation).unsqueeze(0),
                rgb=torch.from_numpy(img_observation).unsqueeze(0)
            )

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
        return super().render()


    def reset_mm(self, seed=0, num_initial_steps=1):
        #self.seed(seed)
        self.reset()
        self.ep_step = 0
        self.ep_reward = 0.

        super().reset()

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
            obs, _, _, _, info = self.step_mm([0.]*6)

        return obs, info

    def close(self):
        super().close()

    def get_state(self):
        return torch.from_numpy(self._get_obs()).unsqueeze(0)

