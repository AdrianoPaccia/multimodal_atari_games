import os
import random
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import numpy as np
import torch
from multimodal_atari_games.multimodal_atari_games.noise.image_noise import ImageNoise
from gym import spaces
from PIL import Image
import copy
from dm_control import suite
from dm_control.suite.wrappers import pixels
os.environ["MUJOCO_GL"] = "egl"

class CheetahImageConfiguration:

    def __init__(
            self,
            render_mode='rgb_array',
            #ram_noise_generator=RamNoise([],0.0, game='pong'),
            image_noise_generator=ImageNoise(noise_types=[], game='cheetah'),
            max_episode_steps=300,
            noise_frequency=0.0
    ):

        #super().__init__(render_mode=render_mode)
        #self.ram_noise_generator=ram_noise_generator
        self.image_noise_generator=image_noise_generator
        self.max_episode_steps = max_episode_steps
        self.noise_frequency = noise_frequency
        self.ep_reward = 0.
        self.device = torch.device('cpu')
        self.obs_modes = ['state','rgb']

        #env setup
        env_ = suite.load('cheetah', 'run')
        self.env = pixels.Wrapper(
            env_,
            pixels_only=False,
            render_kwargs={'height': 100, 'width': 100, 'camera_id': 0},
            observation_key='rgb',
        )
        img_shape = self.env.observation_spec()['rgb'].shape
        self.single_state_shape = (sum(x for x in self.env.observation_spec()['position'].shape + self.env.observation_spec()['velocity'].shape),)
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

        observation, reward, done, truncated, info = self.step(a)

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
            if True:
                observation['rgb'] = self.image_noise_generator.get_observation(observation['rgb'])
            else:
                pass

        state = np.concatenate([v for k, v in observation.items() if k != 'rgb'], axis=-1)
        obs = dict(
            state=torch.from_numpy(state).unsqueeze(0),
            rgb=torch.from_numpy(observation['rgb'].copy()).unsqueeze(0)
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
        return self.env.render(mode='human')

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
            obs, _, _, _, info = self.step_mm([0.]*6)

        return obs, info

    def close(self):
        super().close()

    def get_state(self):
        return torch.from_numpy(self.env._env.physics.state()[1:]).unsqueeze(0)



'''module load Anaconda
conda activate base
source /proj/rep-learning-robotics/users/x_adrpa/MRL/bin/activate

#wandb
export WANDB_API_KEY=42ed698c28ccf287342f44e6978918b9c9812308

cd /proj/rep-learning-robotics/users/x_adrpa/multimodal_representation_for_RL/baselines/sac_TWN
export PYTHONPATH=/proj/rep-learning-robotics/users/x_adrpa/multimodal_representation_for_RL/
export MUJOCO_GL=egl
python sac.py --env=cheetah --num_envs=1 --num_eval_envs=1 --utd=0.5 --buffer_size=30_000 --total_timesteps=1_000_000 --eval_freq=10_000 --no-save_checkpoint --save_model --seed=1 --fusion_strategy=None --gamma=0.9 --batch_size=64 --modes=state+rgb --buffer_device=cuda --sim_device=cuda --eval_freq=10_000 --learning_starts=3000  --training_freq=3000 --wandb_group=berzelius_prove --wandb_name=cheetah_state+rgb --num_eval_episodes=3 --eval_noise_types=gaussian_noise --eval_noise_freq=0.0 --autotune --track

'''