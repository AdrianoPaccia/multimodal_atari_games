import torch
import argparse
available_noises = ['nonoise', 'gaussian_noise', 'partial_obs', 'background_noise', 'sensor_fail', 'random_obs']
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
import os
def build_environment(game):

    if game=='pendulum':
        from multimodal_atari_games.build import build_env_pendulum
        return build_env_pendulum(config=SimpleNamespace(**{
            'max_episode_length': 200,
            'original_frequency': 440.0,
            'sound_receivers': ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'],
            'sound_velocity': 20.0,
            'seed':0
            })
        )

    elif game=='cheetah' or game=='humanoid':
        from multimodal_atari_games.build import build_env_mujoco
        return build_env_mujoco(game, max_episode_steps=300)

    elif game.startswith('antmaze') or game.startswith('pointmaze') or game.startswith('fetch'):
        from multimodal_atari_games.build import build_env_robotics
        return build_env_robotics(game,**{'reward_type':'dense'})
    else:
        raise ValueError(f'{game} is not a valid game.')


class OrnsteinUhlenbeckProcess:
    def __init__(self, action_space, theta=0.02, sigma=0.2):
        self.theta = theta
        self.sigma = sigma
        self.action_space = action_space
        self.reset()

    def reset(self):
        self.action = self.action_space.sample()

    def sample(self):
        act = self.action_space.sample()
        self.action = (1 - self.theta) * self.action + self.sigma * act
        self.action = np.clip(self.action, self.action_space.low, self.action_space.high)
        return self.action



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multimodal RL")
    parser.add_argument("--game", type=str, help="which game to use", default='pendulum',
                        choices=['pendulum', 'cheetah', 'humanoid', 'pointmaze', 'fetch_reach', 'fetch_push'])
    parser.add_argument("-num", "--steps_number", type=int, help="How many steps to collect trajectories.", default=3000)
    args = parser.parse_args()

    env = build_environment(game=args.game)


    oup = OrnsteinUhlenbeckProcess(env.action_space)

    dataset = {k:[] for k in env.obs_modes}
    global_step = 0
    pbar = tqdm(total=args.steps_number)
    while global_step < args.steps_number:
        done = False
        obs,_ = env.reset_mm(seed=0)
        oup.reset()
        while not done and global_step < args.steps_number:
            obs, _, _, done, _ = env.step_mm(oup.sample())
            for k,v in obs.items():
                dataset[k].append(obs[k])
            global_step += 1
            pbar.update()
    pbar.close()

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'multimodal_atari_games/noise/offline_trajectories')
    os.makedirs(folder, exist_ok=True)
    data = {k: torch.cat(v).cpu().numpy() for k,v in dataset.items()}

    np.savez(os.path.join(folder, f'{args.game}.npz'), **data)

