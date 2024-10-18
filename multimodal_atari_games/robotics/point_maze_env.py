import os
import random
import numpy as np
from matplotlib import pyplot as plt
from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise
from PIL import Image
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper

env_ids = {
    'small':'PointMaze_UMaze-v3',
    'medium':'PointMaze_Medium-v3',
    'large':'PointMaze_Large-v3',
}

class PointMazeImageConfiguration:

    def __init__(
            self,
            render_mode='rgb_array',
            image_noise_generator=ImageNoise(noise_types=[], frequency=0.0, game='pointmaze_medium'),
            max_episode_steps=300,
            map_size='medium',
            reward_type='dense'

    ):
        if map_size not in ['small', 'medium', 'large']:
            raise ValueError('Map size must be either "small", "medium" or "large"')
        if reward_type not in ['sparse', 'dense']:
            raise ValueError('Reward type must be either "sparse" or "dense"')

        if render_mode is None:
            self.env = gym.make(env_ids[map_size],render_mode=None, reward_type=reward_type)
        else:
            self.env = PixelObservationWrapper(
                gym.make(env_ids[map_size], render_mode="rgb_array", reward_type=reward_type),
                pixels_only=False)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.image_noise_generator=image_noise_generator
        self.max_episode_steps=max_episode_steps

    def step(self, a):
        obs, reward, done, info, _ = self.env.step(a)

        if self.env_step>=self.max_episode_steps:
            done=True

        self.env_step+=1

        state = np.concatenate([v for k,v in obs.items() if not k =='pixels'])

        # get image observation
        if self.env.render_mode == 'rgb_array':
            config_obs = obs['observation']
            image_observation = obs['pixels']
            img = Image.fromarray(np.uint8(image_observation))
            img = img.resize((200, 200))
            image_observation = np.array(img)
            if random.random() < self.image_noise_generator.frequency:
                image_observation = self.image_noise_generator.get_observation(image_observation)
            # get noisy config observation
            # if random.random() < self.conf_noise_generator.frequency:
            #    config_obs = self.conf_noise_generator.get_observation(config_obs)
            return (image_observation, config_obs), reward, done, info, state
        else:
            return obs, reward, done, info, state

    def render(self):
        return self.env.render()


    def reset(self, num_initial_steps=1):
        obs, _ = self.env.reset()
        self.env_step=0

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
            obs, _, _, _, true_state = self.step([0.]*2)

        return obs, true_state

    def close(self):
        self.env.close()


if __name__ == '__main__':
    from utils.exploration import OrnsteinUhlenbeckProcess
    n_episodes = 5
    env = PointMazeImageConfiguration(
        render_mode='rgb_array',
        image_noise_generator=ImageNoise(game='pointmaze_medium', noise_types=['nonoise'], frequency=0.0),
        map_size='medium',reward_type='sparse'
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

        while not done and steps < 400:
            #action = env.action_space.sample()
            action = oup.sample()
            (image, conf_obs), reward, done, info, state = env.step(action)
            conf.append(conf_obs)
            images.append(image)
            states.append(state)
            #print(reward)
            #plt.imshow(image)
            #plt.show()
            steps+=1
    dio = 0