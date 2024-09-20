

def build_env(game, config=None, obs1_noise_freq=0.0, obs2_noise_freq=0.0, obs1_noises:list=['nonoise'], obs2_noises:list=['nonoise'], render=False):

    if game=='pendulum_multimodal':
        return build_env_pendulum(
            config=config, noise_freq=obs1_noise_freq, image_noises=obs1_noises, sound_noises=obs2_noises, render=render
        )

    elif game=='pong':
        return build_env_atari(
            game=game, image_noise_freq=obs1_noise_freq, ram_noise_freq=obs2_noise_freq,
            image_noises=obs1_noises, ram_noises=obs2_noises, render=render
        )

    elif game=='cheetah' or game=='humanoid':
        return build_env_mujoco(
            game=game, image_noise_freq=obs1_noise_freq, conf_noise_freq=obs2_noise_freq,
            image_noises=obs1_noises, conf_noises=obs2_noises, max_episode_steps=config.max_episode_steps, render=render
        )

    elif game.startswith('antmaze') or game.startswith('pointmaze') or game.startswith('fetch'):
        return build_env_robotics(
            game=game, image_noise_freq=obs1_noise_freq, conf_noise_freq=obs2_noise_freq,
            image_noises=obs1_noises, conf_noises=obs2_noises, max_episode_steps=config.max_episode_steps,
            map_size=config.map_size, reward_type=config.reward_type, render=render)
    else:
        raise ValueError(f'{game} is not a valid game.')


def build_env_pendulum(config, noise_freq=0.0,noise_types:list=['nonoise'], render=False):

    import multimodal_atari_games.multimodal_atari_games.pendulum.pendulum_env as ps
    from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, SoundNoise

    return ps.PendulumSound(
        original_frequency=config.original_frequency,
        sound_vel=config.sound_velocity,
        sound_receivers=[
            ps.SoundReceiver(ps.SoundReceiver.Location[ss])
            for ss in config.sound_receivers
        ],
        image_noise_generator=ImageNoise(
            noise_types=noise_types,
            game='pendulum'),
        sound_noise_generator=SoundNoise(
            noise_types=noise_types,
            game='pendulum'),
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

def build_env_mujoco(game, noise_freq=0.0, noise_types:list=['nonoise'], max_episode_steps=1000, render=False, **kwargs):
    from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise

    if game=='cheetah':
        from multimodal_atari_games.multimodal_atari_games.mujoco.cheetah_env import CheetahImageConfiguration

        return CheetahImageConfiguration(
                render_mode='rgb_array', #None,
                image_noise_generator=ImageNoise(game='cheetah', noise_types=noise_types),
                max_episode_steps=max_episode_steps,
                noise_frequency=noise_freq
                #ram_noise_generator=RamNoise(['random_obs'], 1.0),
            )


    elif game=='humanoid':
        from multimodal_atari_games.multimodal_atari_games.mujoco.humanoid_env import HumanoidImageConfiguration

        return HumanoidImageConfiguration(
            render_mode='rgb_array', #None,
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
