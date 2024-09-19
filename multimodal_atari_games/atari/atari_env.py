import os
from gym.envs.atari.atari_env import AtariEnv
import random
import numpy as np
from matplotlib import pyplot as plt
from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise
from multimodal_atari_games.multimodal_atari_games.noise.ram_noise import RamNoise


class AtariImageRam(AtariEnv):

    def __init__(
            self,
            game='pong',
            mode=None,
            difficulty=None,
            ram_noise_generator=RamNoise(noise_types=[],game='pong'),
            image_noise_generator=ImageNoise(noise_types=[], game='pong'),
            noise_frequency=0.0
    ):
        super().__init__(game=game, mode=mode, difficulty=difficulty, obs_type='ram')
        self.ram_noise_generator=ram_noise_generator
        self.image_noise_generator=image_noise_generator
        self.noise_frequency= noise_frequency
    def step(self, a):
        ram_observation, reward, done, info = super().step(a)

        image_observation = super().render(mode='rgb_array')
        if random.random() < self.noise_frequency:
            if bool(random.getrandbits(1)):
                # get image observation
                image_observation = self.image_noise_generator.get_observation(image_observation)
            else:
                # get noisy ram observation
                ram_observation = self.ram_noise_generator.get_observation(ram_observation)

        return (image_observation,ram_observation), reward, done, info, None

    def render(self, mode='human'):
        if mode == "human":
            super().render()
        return super().render(mode='rgb_array')


    def reset(self, num_initial_steps=1):
        ram = super().reset()

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
            (image, ram), _, _, _, true_state = self.step(0)

        return (image, ram),true_state

    def close(self, out=None):
        super().close()


if __name__ == '__main__':
    game='pong'
    n_episodes = 5
    env = AtariImageRam(
        game=game,
        difficulty=None,
        image_noise_generator=ImageNoise(['random_obs', 'salt_pepper_noise'],  noise_types=1.0),
        ram_noise_generator=RamNoise(['random_obs'], noise_types=1.0)
    )
    rams = []
    images = []
    for _ in range(n_episodes):
        done = False
        steps = 0
        (ram, image), _ = env.reset()
        while not done and steps < 1500:
            action = env.action_space.sample()
            (ram, image), reward, done, info,_ = env.step(action)
            rams.append(ram)
            images.append(image)
            plt.imshow(image)
            plt.show()
            steps+=1
    dir = os.path.join(os.getcwd(),f'offline_trajectories/{game}')
    os.makedirs(dir, exist_ok=True)

    np.savez(os.path.join(dir,f'images.npz'),
            images=np.stack(images),allow_pickle=True
    )
    np.savez(os.path.join(dir,f'rams.npz'),
            rams=np.stack(rams),allow_pickle=True
    )

    env.close()