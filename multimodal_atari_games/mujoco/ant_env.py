import os
import random
import numpy as np
import torch
from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, StateNoise, DepthNoise
from gym import spaces
from dm_control import suite
from dm_control.suite.wrappers import pixels
os.environ["MUJOCO_GL"] = "egl"



class AntImageConfiguration:

    def __init__(
            self,
            noise_generators = {
                'rgb': ImageNoise(noise_types=[], game='ant'),
                'depth': DepthNoise(noise_types=[], game='ant'),
                'state': StateNoise(noise_types=[], game='ant'),

            },
            max_episode_steps=200,
            noise_frequency=0.0
    ):
        self.noise_generators = noise_generators
        self.max_episode_steps = max_episode_steps
        self.noise_frequency = noise_frequency
        self.ep_reward = 0.
        self.device = torch.device('cpu')
        self.obs_modes = ('state', 'rgb', 'depth')
        self.n_noisy_obs = 1 #number of obs to inject noise to

        #built env
        env_ = suite.load('quadruped', 'escape')

        #wrapper for image observations
        env = pixels.Wrapper(
            env_,
            pixels_only=False,
            render_kwargs={'height': 100, 'width': 100, 'camera_id': 0},
            observation_key='rgb',
        )

        #wrapper for depth observations
        self.env = pixels.Wrapper(
            env,
            pixels_only=False,
            render_kwargs={'height': 100, 'width': 100, 'camera_id': 0, 'depth':True},
            observation_key='depth',
        )

        self.state_keys = ['egocentric_state', 'torso_velocity', 'torso_upright', 'imu', 'force_torque', 'origin', 'rangefinder']
        self.single_state_shape = (101,)

        self.state_space = spaces.Box(low=-8., high=8., shape=self.single_state_shape)
        self.single_observation_space_mm = spaces.Tuple([
            self.state_space,  # state
            spaces.Box(low=0, high=255, shape=self.env.observation_spec()['rgb'].shape),  # image
            spaces.Box(low=0, high=100, shape=self.env.observation_spec()['depth'].shape),  # depth
        ])

        self.observation_space_mm = spaces.Tuple([
            spaces.Box(low=-8., high=8., shape=(1,)+self.single_state_shape),  # state
            spaces.Box(low=0, high=255, shape=(1,)+self.env.observation_spec()['rgb'].shape),  # image
            spaces.Box(low=0, high=100, shape=(1,)+self.env.observation_spec()['depth'].shape),  # depth
        ])

        self.single_action_space = spaces.Box(
            low=self.env.action_spec().minimum[0],
            high=self.env.action_spec().maximum[0],
            shape=self.env.action_spec().shape
        )
        self.action_space = self.single_action_space


    def step(self, a):
        timestep = self.env.step(a)
        truncated = self.env._step_count > self.max_episode_steps
        self.ep_reward += timestep.reward
        return timestep.observation, timestep.reward, timestep.last(), truncated, {}

    def step_mm(self, a):

        if torch.is_tensor(a):
            a = a.numpy().reshape(-1)

        self.observation, reward, done, truncated, info = self.step(a)

        if self.env._step_count >= self.max_episode_steps:
            truncated = True

        #assemble the obs
        obs = dict(
            state=np.concatenate([self.observation[k].reshape(-1) for k in self.state_keys]),
            rgb=self.observation['rgb'].copy(),
            depth=self.observation['depth'].copy()
        )

        # inject noise
        if random.random() < self.noise_frequency:
            for m in random.sample(list(self.noise_generators.keys()), self.n_noisy_obs):
                obs[m] = self.noise_generators[m].get_observation(obs[m])

        obs = {m: torch.from_numpy(o).unsqueeze(0) for m,o in obs.items()}
        reward = torch.tensor([reward]).unsqueeze(0)
        done = torch.tensor([done]).unsqueeze(0)
        truncated = torch.tensor([truncated]).unsqueeze(0)

        info = {
            'elapsed_steps': torch.tensor([self.env._step_count]),
            'episode': {'r': torch.tensor([self.ep_reward])}
        }

        if done or truncated:
            info['final_info'] = {
                'elapsed_steps': torch.tensor([self.env._step_count]),
                'episode': {
                    'r': torch.tensor([self.ep_reward]),
                    '_r': torch.tensor([True])
                },
            }
            info['_final_info'] = torch.tensor([True])
            info['final_observation'] = obs

        return obs, reward, done, truncated, info

    def render(self):
        return self.env._env.physics.render()

    def reset(self):
        self.ep_reward = 0.
        return self.env.reset()

    def reset_mm(self, seed=0, num_initial_steps=1):
        #self.seed(seed)
        self.reset()

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
            obs, _, _, _, info = self.step_mm([0.]*sum(self.env.action_spec().shape))

        return obs, info

    def close(self):
        self.env.close()

    def get_state(self):
        s = np.concatenate([self.observation[k].reshape(-1) for k in self.state_keys])
        return torch.from_numpy(s).unsqueeze(0)
