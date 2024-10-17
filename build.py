def build_env_pendulum(
        config,
        noise_modes: list = [],
        noise_freq: float = 0.0,
        noise_types: list = ['nonoise'],
        render=False):
    """
    Builds the pendulum environment.
    :param config: sound configuration NameSpace (original_frequency, sound_velocity, sound_receivers)
    :param noise_modes: list = modes affected by noise
    :param noise_freq: float = frequency of noisy observations
    :param noise_types: list = noises applied to the noise_modes
    :param render: bool = True (for rendering the obs)
    :return: pendulum multimodal env
    """
    import multimodal_atari_games.multimodal_atari_games.pendulum.pendulum_env as ps
    from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, SoundNoise, StateNoise
    noise_generators = {}
    if 'rgb' in noise_modes:
        noise_generators['rgb'] = ImageNoise(noise_types=noise_types,game='pendulum',
                           **{'bounds': (config.low_bounds['rgb'], config.high_bounds['rgb'])})
    if 'sound' in noise_modes:
        noise_generators['sound'] = SoundNoise(noise_types=noise_types, game='pendulum',
                           **{'bounds': (config.low_bounds['sound'], config.high_bounds['sound'])})
    if 'state' in noise_modes:
        noise_generators['state'] = StateNoise(noise_types=noise_types, game='pendulum',
                       **{'low_bounds': (config.low_bounds['state'], config.high_bounds['state'])})

    return ps.PendulumSound(
        original_frequency=config.original_frequency,
        sound_vel=config.sound_velocity,
        sound_receivers=[
            ps.SoundReceiver(ps.SoundReceiver.Location[ss])
            for ss in config.sound_receivers
        ],
        noise_generators=noise_generators,
        noise_frequency=noise_freq,
        rendering_mode='human' if render else 'rgb_array',
    )




def build_env_atari(game, noise_freq=0.0, noise_types:list=['nonoise'], render=False):
    from multimodal_atari_games.multimodal_atari_games.atari.atari_env import AtariImageRam
    from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise
    from multimodal_atari_games.multimodal_atari_games.noise.ram_noise import RamNoise
    return AtariImageRam(
        game=game,
        difficulty=None,
        image_noise_generator=ImageNoise(noise_types=noise_types, game=game),
        ram_noise_generator=RamNoise(noise_types=noise_types, game=game),
        noise_frequency=noise_freq
    )

def build_env_mujoco(
        game,
        noise_freq=0.0,
        noise_types:list=['nonoise'],
        noise_modes:list=[],
        max_episode_steps=1000,
        **kwargs):

    from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, StateNoise, DepthNoise
    noise_generators = {}
    if 'rgb' in noise_modes:
        noise_generators['rgb'] = ImageNoise(noise_types=noise_types, game=game,
                                             **{'bounds': (kwargs['low_bounds']['rgb'], kwargs['high_bounds']['rgb'])})
    if 'depth' in noise_modes:
        noise_generators['depth'] = DepthNoise(noise_types=noise_types, game=game,
                                               **{'bounds': (kwargs['low_bounds']['depth'], kwargs['high_bounds']['depth'])})
    if 'state' in noise_modes:
        noise_generators['state'] = StateNoise(noise_types=noise_types, game=game,
                                               **{'low_bounds': (
                                               kwargs['low_bounds']['state'], kwargs['high_bounds']['state'])})

    if game == 'cheetah':
        from multimodal_atari_games.multimodal_atari_games.mujoco.cheetah_env import CheetahImageConfiguration

        return CheetahImageConfiguration(
            noise_generators=noise_generators,
            max_episode_steps=max_episode_steps,
            noise_frequency=noise_freq
            )

    elif game=='ant':
        from multimodal_atari_games.multimodal_atari_games.mujoco.ant_env import AntImageConfiguration

        return AntImageConfiguration(
            noise_generators=noise_generators,
            max_episode_steps=max_episode_steps,
            noise_frequency=noise_freq
            )


    elif game=='humanoid':
        from multimodal_atari_games.multimodal_atari_games.mujoco.humanoid_env import HumanoidImageConfiguration

        return HumanoidImageConfiguration(
            image_noise_generator=ImageNoise(game='humanoid', noise_types=noise_types),
            max_episode_steps=max_episode_steps,
            noise_frequency=noise_freq
            # ram_noise_generator=RamNoise(['random_obs'], 1.0)
        )

    else:
        raise ValueError(f'{game} is not a valid game (cheetah, humanoid)!')



def build_env_robotics(game, noise_freq=0.0, noise_types:list=['nonoise'], max_episode_steps=1000, render=False, **kwargs):
    from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise

    if game=='fetch_reach':
        from multimodal_atari_games.multimodal_atari_games.robotics.fetch_env import FetchReachImageConfiguration
        return FetchReachImageConfiguration(
            render_mode='rgb_array',
            image_noise_generator=ImageNoise(game='fetch_reach', noise_types=noise_types),
            max_episode_steps=max_episode_steps,
            #ram_noise_generator=RamNoise(['random_obs'], 1.0),
            reward_type=kwargs['reward_type']
        )
    if game=='fetch_push':
        from multimodal_atari_games.multimodal_atari_games.robotics.fetch_env import FetchPushImageConfiguration
        return FetchPushImageConfiguration(
            render_mode='rgb_array',
            image_noise_generator=ImageNoise(game='fetch_push', noise_types=noise_types),
            max_episode_steps=max_episode_steps,
            #ram_noise_generator=RamNoise(['random_obs'], 1.0),
            reward_type=kwargs['reward_type']
        )

    elif game.startswith('antmaze'):
        from multimodal_atari_games.multimodal_atari_games.robotics.ant_maze_env import AntMazeImageConfiguration
        return AntMazeImageConfiguration(
            render_mode='rgb_array',
            image_noise_generator=ImageNoise(game=game, noise_types=noise_types),
            max_episode_steps=max_episode_steps,
            # ram_noise_generator=RamNoise(['random_obs'], 1.0)
            map_size=kwargs['map_size'],
            reward_type=kwargs['reward_type']
        )

    elif game.startswith('pointmaze'):
        from multimodal_atari_games.multimodal_atari_games.robotics.point_maze_env import PointMazeImageConfiguration
        return PointMazeImageConfiguration(
            render_mode='rgb_array',
            image_noise_generator=ImageNoise(game=game, noise_types=noise_types),
            max_episode_steps=max_episode_steps,
            # ram_noise_generator=RamNoise(['random_obs'], 1.0)
            map_size=kwargs['map_size'],
            reward_type=kwargs['reward_type']
        )
    else:
        raise ValueError(f'{game} is not a valid game (fetch_reach, fetch_psuh, antmaze, pointmaze)!')
