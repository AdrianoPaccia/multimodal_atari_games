from multimodal_atari_games.multimodal_atari_games.rl_zoo3.base_env import BaseRLzoo3Env
import yaml
from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, StateNoise
from os.path import dirname, abspath

with open(dirname(abspath(__file__)) + '/configurations.yaml', "r") as file:
    config = yaml.safe_load(file)
noise_generators = {
    'rgb': ImageNoise(noise_types=[], game='cheetah'),
    'state': StateNoise(noise_types=[], game='cheetah'),
}

class RLzoo3Env(BaseRLzoo3Env):

    def __init__(
            self,
            game: str,
            noise_generators=noise_generators,
            noise_frequency=0.0,
            n_envs=1,
            seed=0,
    ):

        super().__init__(
            env_id=config[game]['id'],
            args=config[game]['args'],
            max_episode_steps=config[game]['max_episode_steps'],
            noise_generators=noise_generators,
            noise_frequency=noise_frequency,
            **dict(num_envs=n_envs, seed=seed)
        )

if __name__ == '__main__':
    env = RLzoo3Env(
        game='fetch_push',
    )
    env.show_description()
