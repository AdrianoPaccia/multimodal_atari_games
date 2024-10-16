import os
import random
import numpy as np
import torch
from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, StateNoise
from gym import spaces
from dm_control import suite
from dm_control.suite.wrappers import pixels
os.environ["MUJOCO_GL"] = "egl"



class HumanoidImageConfiguration:

    def __init__(
            self,
            render_mode='rgb_array',
            state_noise_generator=StateNoise(game='cheetah', noise_types=[]),
            image_noise_generator=ImageNoise(noise_types=[], game='cheetah'),
            max_episode_steps=300,
            noise_frequency=0.0
    ):

        self.state_noise_generator = state_noise_generator
        self.image_noise_generator = image_noise_generator
        self.max_episode_steps = max_episode_steps
        self.noise_frequency = noise_frequency
        self.ep_reward = 0.
        self.device = torch.device('cpu')
        self.obs_modes = ['state','rgb']
        self.state = None

        #env setup
        env_ = suite.load('humanoid', 'run')
        self.env = pixels.Wrapper(
            env_,
            pixels_only=False,
            render_kwargs={'height': 100, 'width': 100, 'camera_id': 0},
            observation_key='rgb',
        )
        img_shape = self.env.observation_spec()['rgb'].shape
        self.single_state_shape = (67,) #(55,)
        self.state_keys = ['joint_angles', 'head_height', 'extremities', 'torso_vertical', 'com_velocity', 'velocity']

        self.state_space = spaces.Box(low=-8., high=8., shape=self.single_state_shape)
        self.single_observation_space_mm = spaces.Tuple([
            self.state_space,  # state
            spaces.Box(low=0, high=255, shape=img_shape),  # image
        ])

        self.observation_space_mm = spaces.Tuple([
            spaces.Box(low=-8., high=8., shape=(1,)+self.single_state_shape),  # state
            spaces.Box(low=0, high=255, shape=(1,)+img_shape),  # image
        ])

        act_spec = self.env.action_spec()
        self.single_action_space = spaces.Box(low=act_spec.minimum[0], high=act_spec.maximum[0], shape=act_spec.shape)
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
        image = self.observation['rgb']
        state = np.concatenate([self.observation[k].reshape(-1) for k in self.state_keys])

        if self.env._step_count >= self.max_episode_steps:
            truncated = True

        reward = torch.tensor([reward]).unsqueeze(0)
        done = torch.tensor([done]).unsqueeze(0)
        truncated = torch.tensor([truncated]).unsqueeze(0)

        info = {
            'elapsed_steps': torch.tensor([self.env._step_count]),
            'episode': {'r': torch.tensor([self.ep_reward])}
        }

        if random.random() < self.noise_frequency:
            if random.random() < 0.5:
                image = self.image_noise_generator.get_observation(image)
            else:
                state = self.state_noise_generator.get_observation(state)

        obs = dict(
            state=torch.from_numpy(state).unsqueeze(0),
            rgb=torch.from_numpy(image.copy()).unsqueeze(0)
        )

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
        return torch.from_numpy(np.concatenate([self.observation[k].reshape(-1) for k in self.state_keys])).unsqueeze(0)
        #return torch.from_numpy(self.env._env.physics.get_state()).unsqueeze(0)
