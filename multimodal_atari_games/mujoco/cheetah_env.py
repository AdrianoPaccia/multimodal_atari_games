import os
import random
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import numpy as np
from matplotlib import pyplot as plt
from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise
from utils.exploration import OrnsteinUhlenbeckProcess

class CheetahImageConfiguration(HalfCheetahEnv):

    def __init__(
            self,
            render_mode=None,
            #ram_noise_generator=RamNoise([],0.0, game='pong'),
            image_noise_generator=ImageNoise(noise_types=[], frequency=0.0, game='cheetah'),
            max_episode_steps=300
    ):
        super().__init__(render_mode=render_mode)
        #self.ram_noise_generator=ram_noise_generator
        self.image_noise_generator=image_noise_generator
        self.max_episode_steps = max_episode_steps

    def step(self, a):
        config_obs, reward, done, truncated, info = super().step(a)

        if self.env_step >= self.max_episode_steps:
            done = True

        # get noisy config observation
        #if random.random() < self.ram_noise_generator.frequency:
        #    ram_observation = self.ram_noise_generator.get_observation(ram_observation)

        # get image observation
        try:
            image_observation = super().render()[140:480,80:420]
            if random.random() < self.image_noise_generator.frequency:
                image_observation = self.image_noise_generator.get_observation(image_observation)
            self.env_step +=1
            return (image_observation,config_obs), reward, done, info, config_obs
        except:
            return (None,config_obs), reward, done, info, config_obs

    def render(self):
        return super().render()


    def reset(self, num_initial_steps=1):
        config_obs, _ = super().reset()
        self.env_step = 0

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
            obs, _, _, _, true_state = self.step([0.]*6)

        return obs, true_state

    def close(self):
        super().close()


if __name__ == '__main__':
    n_episodes = 5
    env = CheetahImageConfiguration(
        render_mode='rgb_array',
        image_noise_generator=ImageNoise(game='cheetah', noise_types=['random_obs'], frequency=1.0),
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

        while not done and steps < 200:
            #action = env.action_space.sample()
            action = oup.sample()
            (image, conf_obs), reward, done, info, state = env.step(action)
            conf.append(conf_obs)
            images.append(image)
            states.append(state)
            plt.imshow(image)
            plt.show()
            steps+=1

    dio = 0

