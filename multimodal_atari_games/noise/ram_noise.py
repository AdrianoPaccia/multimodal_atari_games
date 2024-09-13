import numpy as np
import random
import os
import yaml

class RamNoise:
    def __init__(self,  game:str, noise_types: list):
        self.noise_types = noise_types
        with open(os.path.join(os.path.dirname(__file__), f'config/{game}.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)['rams']
        self.game=game
        if not set(noise_types).issubset(set(self.config['available_noises'])):
            raise ValueError("Noise types not supported")

    def get_observation(self, ram):
        noise = random.choice(self.noise_types)
        return self.apply_noise(noise, ram)

    def apply_noise(self, noise_type: str, ram):
        if noise_type == 'random_obs':
            noisy_ram = self.get_random_observation()
        elif noise_type == 'nonoise':
            noisy_ram = ram
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return noisy_ram

    def apply_random_noise(self, ram):
        noise_type = random.choice(list(self.config.keys()))
        return self.apply_noise(noise_type, ram), noise_type

    def apply_all_noises(self, sound):
        noisy_sounds = []
        for noise_type in self.noise_types:
            noisy_sounds.append(self.apply_noise(noise_type, sound))
        return noisy_sounds, self.noise_types

    def get_random_observation(self):
        rams = np.load(os.path.join(os.path.dirname(__file__),f'offline_trajectories/{self.game}/rams.npz'), allow_pickle=True)['rams']
        i_rand = random.randint(0, rams.shape[0] - 1)
        return rams[i_rand]


