import os
import random
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
import numpy as np
from matplotlib import pyplot as plt
from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise
from gym import spaces
from PIL import Image
import copy
import torch

class HumanoidImageConfiguration(HumanoidEnv):

    def __init__(
            self,
            render_mode='rgb_array',
            #ram_noise_generator=RamNoise([],0.0, game='pong'),
            image_noise_generator=ImageNoise(noise_types=[], frequency=0.0, game='cheetah'),
            max_episode_steps=300,
            noise_frequency=0.0
    ):
        super().__init__(render_mode=render_mode)
        #self.ram_noise_generator=ram_noise_generator
        self.image_noise_generator=image_noise_generator
        self.max_episode_steps = max_episode_steps
        self.noise_frequency = noise_frequency
        self.ep_step, self.ep_reward = 0, 0
        self.device = torch.device('cpu')
        self.obs_modes = ['state', 'rgb']
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

        done = done or truncated
        if self.env_step >= self.max_episode_steps:
            done = True

        self.env_step += 1
        img_observation = super().render()[180:480, 100:400]
        if random.random() < self.noise_frequency:
            img_observation = self.image_noise_generator.get_observation(img_observation)

        rgb = Image.fromarray(img_observation)
        img_observation = np.array(rgb.resize((100, 100)))


        # get noisy config observation
        #if random.random() < self.ram_noise_generator.frequency:
        #    ram_observation = self.ram_noise_generator.get_observation(ram_observation)

        obs = dict(
            state=torch.tensor(ram_observation).unsqueeze(0),
            rgb=torch.from_numpy(img_observation).unsqueeze(0)
        )
        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)
        truncated = torch.tensor(truncated).unsqueeze(0)

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
        return super().render()


    def reset_mm(self, num_initial_steps=1):
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
            obs, _, _, _, info = self.step_mm([0.]*17)

        return obs, info

    def close(self):
        super().close()

    def get_state(self):
        return torch.from_numpy(self._get_obs()).unsqueeze(0)



if __name__ == '__main__':
    from utils.exploration import OrnsteinUhlenbeckProcess
    n_episodes = 5
    env = HumanoidImageConfiguration(
        render_mode='rgb_array',
        image_noise_generator=ImageNoise(game='humanoid', noise_types=['random_obs'], frequency=1.0),
        #image_noise_generator=ImageNoise(game='cheetah', noise_types=['salt_pepper_noise'], frequency=1.0),
        #ram_noise_generator=RamNoise(['random_obs'], 1.0)
    )
    conf, images,states = [], [], []
    oup = OrnsteinUhlenbeckProcess(env.action_space)

    for _ in range(n_episodes):
        done = False
        steps = 0
        (image, conf_obs), _ = env.reset()
        oup.reset()

        while not done and steps < 1000:
            # action = env.action_space.sample()
            action = oup.sample()
            (image, conf_obs), reward, done, info, state = env.step(action)
            conf.append(conf_obs)
            images.append(image)
            states.append(state)
            print(steps, ': ', reward)
            # plt.imshow(image)
            # plt.show()
            steps += 1