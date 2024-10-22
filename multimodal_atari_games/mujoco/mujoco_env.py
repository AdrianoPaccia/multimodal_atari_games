from multimodal_atari_games.multimodal_atari_games.mujoco.base_env import BaseMujocoEnv
import yaml
from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, StateNoise, DepthNoise
from os.path import dirname, abspath

with open(dirname(abspath(__file__)) + '/configurations.yaml', "r") as file:
    config = yaml.safe_load(file)

noise_generators = {
    'rgb': ImageNoise(noise_types=[], game='cheetah'),
    'depth': DepthNoise(noise_types=[], game='cheetah'),
    'state': StateNoise(noise_types=[], game='cheetah'),
}
class MujocoEnv(BaseMujocoEnv):

    def __init__(
            self,
            env: str,
            noise_generators = noise_generators,
            noise_frequency = 0.0
    ):
        super().__init__(
            game=config[env]['game'],
            task=config[env]['task'],
            state_keys=config[env]['state_keys'],
            max_episode_steps=config[env]['max_episode_steps'],
            noise_generators=noise_generators,
            noise_frequency=noise_frequency,

        )

if __name__ == '__main__':
    env = MujocoEnv(
        env='ant',
    )
    env.show_description()
