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


def build_env_mujoco(
        game,
        noise_freq=0.0,
        noise_types:list=['nonoise'],
        noise_modes:list=[],
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
                                               **{'bounds': (kwargs['low_bounds']['state'], kwargs['high_bounds']['state'])})

    if game == 'cheetah':
        from multimodal_atari_games.multimodal_atari_games.mujoco.cheetah_env import CheetahImageConfiguration

        return CheetahImageConfiguration(
            noise_generators=noise_generators,
            noise_frequency=noise_freq
            )
    else:
        from multimodal_atari_games.multimodal_atari_games.mujoco.mujoco_env import MujocoEnv
        return MujocoEnv(
            env=game,
            noise_generators=noise_generators,
            noise_frequency=noise_freq
            )



def build_env_robotics(game, noise_freq=0.0, noise_types:list=['nonoise'], max_episode_steps=1000, render=False, **kwargs):
    from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, RamNoise

    if game=='fetch_reach':
        from multimodal_atari_games.multimodal_atari_games.robotics.fetch_env import FetchReachImageConfiguration
        return FetchReachImageConfiguration(
            render_mode='rgb_array',
            image_noise_generator=ImageNoise(game='fetch_reach', noise_types=noise_types),
            ram_noise_generator=RamNoise(game='fetch_reach', noise_types=noise_types),
            reward_type=kwargs['reward_type'],
            noise_frequency=noise_freq
        )
    if game=='fetch_push':
        from multimodal_atari_games.multimodal_atari_games.robotics.fetch_env import FetchPushImageConfiguration
        return FetchPushImageConfiguration(
            render_mode='rgb_array',
            image_noise_generator=ImageNoise(game='fetch_push', noise_types=noise_types),
            ram_noise_generator=RamNoise(game='fetch_push', noise_types=noise_types),
            reward_type=kwargs['reward_type'],
            noise_frequency=noise_freq
        )
    else:
        raise ValueError(f'{game} is not a valid game (fetch_reach, fetch_psuh, antmaze, pointmaze)!')
