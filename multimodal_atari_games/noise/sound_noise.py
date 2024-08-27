import numpy as np
import random
import os
import yaml

class SoundNoise:
    def __init__(self, game:str, noise_types: list=[], frequency:float=0.0):
        self.noise_types = noise_types
        self.frequency = frequency
        self.game = game
        with open(os.path.join(os.path.dirname(__file__), f'config/{game}.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)['sounds']
        self.random_sounds = np.load(os.path.join(os.path.dirname(__file__),f'offline_trajectories/{self.game}.npz'),
                                     allow_pickle=True)['sounds']

        if not set(noise_types).issubset(set(self.config['available_noises'])):
            raise ValueError("Noise types not supported")


    def get_observation(self, sound):
        noise = random.choice(self.noise_types)
        return self.apply_noise(noise, sound)

    def apply_noise(self, noise_type:str, sound):
        snd = sound.copy()
        if noise_type == 'gaussian_noise':
            noisy_sound = self.apply_gaussian_noise(snd)
        elif noise_type == 'white_noise':
            noisy_sound = self.apply_white_noise(snd)
        elif noise_type == 'pink_noise':
            noisy_sound = self.apply_pink_noise(snd)
        elif noise_type == 'brownian_noise':
            noisy_sound = self.apply_brownian_noise(snd)
        elif noise_type == 'blue_noise':
            noisy_sound = self.apply_blue_noise(snd)
        elif noise_type == 'poisson_noise':
            noisy_sound = self.apply_poisson_noise(snd)
        elif noise_type == 'random_obs':
            noisy_sound = self.get_random_observation()
        elif noise_type == 'nonoise':
            noisy_sound = snd
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return noisy_sound

    def apply_random_noise(self, sound):
        noise_type = random.choice(list(self.config.keys()))
        return self.apply_noise(noise_type, sound), noise_type

    def apply_all_noises(self,sound):
        noisy_sounds = []
        for noise_type in self.noise_types:
            noisy_sounds.append(self.apply_noise(noise_type, sound))
        return noisy_sounds, self.noise_types

    #all implemented noises
    def apply_gaussian_noise(self, sound):
        amplitude_noise = np.random.normal(
            self.config['gaussian_noise']['amplitude_mu'],
            self.config['gaussian_noise']['amplitude_std'],
        len(sound)
        )
        frequency_noise = np.random.normal(
            self.config['gaussian_noise']['frequency_mu'],
            self.config['gaussian_noise']['frequency_std'],
            len(sound)
        )
        noisy_sound = [(s[0]+frequency_noise[i], s[1]+amplitude_noise[i]) for i,s in enumerate(sound)]
        return noisy_sound

    def apply_white_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        noisy_sound = sound + noise
        return noisy_sound

    def apply_pink_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        pink_noise = np.convolve(noise, np.ones(10)/10, mode='same')
        noisy_sound = sound + pink_noise
        return noisy_sound

    def apply_brownian_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        brownian_noise = np.cumsum(noise)
        noisy_sound = sound + brownian_noise
        return noisy_sound

    def apply_blue_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        blue_noise = np.cumsum(noise)
        noisy_sound = sound + blue_noise
        return noisy_sound

    def apply_poisson_noise(self, sound):
        mean = self.config['poisson_noise']['mean']
        poisson_noise = np.random.poisson(mean, len(sound))
        noisy_sound = sound + poisson_noise
        return noisy_sound

    def get_random_observation(self):
        i_rand = random.randint(0, self.random_sounds.shape[0] - 1)
        return self.random_sounds[i_rand]
