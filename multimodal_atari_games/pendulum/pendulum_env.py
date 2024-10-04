import copy

import torchvision.transforms
from gym import spaces
import numpy as np
import pickle
from enum import Enum
from gym.envs.classic_control.pendulum import PendulumEnv
from multimodal_atari_games.multimodal_atari_games.noise.noise import ImageNoise, SoundNoise, StateNoise
import random
#import cv2
import PIL.Image as Image
import torch

def modified_doppler_effect(freq, obs_pos, obs_vel, obs_speed, src_pos,
                            src_vel, src_speed, sound_vel):
    # Normalize velocity vectors to find their directions (zero values
    # have no direction).
    if not np.all(src_vel == 0):
        src_vel = src_vel / np.linalg.norm(src_vel)
    if not np.all(obs_vel == 0):
        obs_vel = obs_vel / np.linalg.norm(obs_vel)

    src_to_obs = obs_pos - src_pos
    obs_to_src = src_pos - obs_pos
    if not np.all(src_to_obs == 0):
        src_to_obs = src_to_obs / np.linalg.norm(src_to_obs)
    if not np.all(obs_to_src == 0):
        obs_to_src = obs_to_src / np.linalg.norm(obs_to_src)

    src_radial_vel = src_speed * src_vel.dot(src_to_obs)
    obs_radial_vel = obs_speed * obs_vel.dot(obs_to_src)

    fp = ((sound_vel + obs_radial_vel) / (sound_vel - src_radial_vel)) * freq

    return fp


def inverse_square_law_observer_receiver(obs_pos, src_pos, K=1.0, eps=0.0):
    """
    Computes the inverse square law for an observer receiver pair.
    Follows https://en.wikipedia.org/wiki/Inverse-square_law
    """
    distance = np.linalg.norm(obs_pos - src_pos)
    return K * 1.0 / (distance**2 + eps)


BOTTOM_MARGIN = -2.2
TOP_MARGIN = 2.2
LEFT_MARGIN = 2.2
RIGHT_MARGIN = -2.2

class SoundReceiver(object):
    class Location(Enum):
        LEFT_BOTTOM = 1,
        LEFT_MIDDLE = 2,
        LEFT_TOP = 3,
        RIGHT_TOP = 4,
        RIGHT_MIDDLE = 5,
        RIGHT_BOTTOM = 6,
        MIDDLE_TOP = 7,
        MIDDLE_BOTTOM = 8

    def __init__(self, location):
        self.location = location

        if location == SoundReceiver.Location.LEFT_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, LEFT_MARGIN])
        elif location == SoundReceiver.Location.LEFT_MIDDLE:
            self.pos = np.array([0.0, LEFT_MARGIN])
        elif location == SoundReceiver.Location.LEFT_TOP:
            self.pos = np.array([TOP_MARGIN, LEFT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_TOP:
            self.pos = np.array([TOP_MARGIN, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_MIDDLE:
            self.pos = np.array([0.0, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.MIDDLE_TOP:
            self.pos = np.array([TOP_MARGIN, 0.0])
        elif location == SoundReceiver.Location.MIDDLE_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, 0.0])


import torchvision.transforms as tf
rs = tf.Resize(size=(100,100))

class PendulumSound(PendulumEnv):
    """
    Frame:
    - points stored as (height, weight)
    - positive upwards and left
    Angular velocity:
    - positive is ccw
    """

    def __init__(
            self,
            original_frequency=440.,
            sound_vel=20.,
            sound_receivers=[SoundReceiver(SoundReceiver.Location.RIGHT_TOP)],
            noise_generators={
                'rgb':ImageNoise(noise_types=[], game='pendulum'),
                'sound':SoundNoise(noise_types=[], game='pendulum'),
                'state':StateNoise(noise_types=[], game='pendulum'),

            },
            #image_noise_generator=ImageNoise(noise_types=[], game='pendulum'),
            #sound_noise_generator=SoundNoise(noise_types=[], game='pendulum'),
            noise_frequency=0.0,
            rendering_mode='rgb_array',
            max_steps=200,
            debug=False):
        super().__init__()
        self.original_frequency = original_frequency
        self.sound_vel = sound_vel
        self.sound_receivers = sound_receivers
        self.noise_frequency = noise_frequency
        self.rendering_mode = rendering_mode
        self._debug = debug
        self.max_steps = max_steps
        self.ep_step = 0
        self.device = torch.device('cpu')
        self.img_trans = torchvision.transforms.Resize((100,100))
        self.noise_generators = noise_generators
        self.n_noisy_obs = len(self.noise_generators)-1
        #only one obs is not affected by noise


        #spaces
        self.obs_modes = ['state','rgb', 'sound']
        img_shape = (100, 100, 3)
        sound_shape = (6,) #(3, 2)
        self.state_space = self.observation_space
        self.single_state_shape = self.observation_space.shape
        self.observation_space_mm = spaces.Tuple([
            self.state_space, #state
            spaces.Box(low=0, high=255, shape=img_shape), #image
            spaces.Box(low=-np.inf, high=np.inf, shape=sound_shape) #sound
        ])
        self.single_observation_space_mm = self.observation_space_mm
        self.single_action_space = copy.deepcopy(self.action_space)

        #rendering stuff
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.reset()


    def step_mm(self, a):
        if torch.is_tensor(a):
            a = a.numpy().reshape(-1)

        state, reward, done, info = super().step(a)
        done = self.ep_step > self.max_steps
        self.ep_step += 1
        self.ep_reward += reward

        x, y, thdot = state

        #get the observations and build the dict
        obs = dict(
            state=state,
            sound=self.sounder_(x, y, thdot),
            rgb=self.render_(np.arctan2(y, x), mode=self.rendering_mode)[100:400, 100:400, :]
        )

        # inject noise
        if random.random() < self.noise_frequency:
            for m in random.sample(list(self.noise_generators.keys()), self.n_noisy_obs):
                obs[m] = self.noise_generators[m].get_observation(obs[m])

        #process the obs:
        obs['sound'] /= np.array([600, 1.]) #norm the sound
        rgb = Image.fromarray(obs['rgb'])
        obs['rgb'] = np.array(rgb.resize((100,100))) #resize the image
        obs = {m: torch.from_numpy(o).unsqueeze(0) for m,o in obs.items()}
        reward = torch.tensor(reward).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)
        info = {
            'elapsed_steps': torch.tensor([self.ep_step]),
            'episode': {'r': torch.tensor([self.ep_reward])}
        }

        if done:
            info['final_info'] = {
                'elapsed_steps': torch.tensor([self.ep_step]),
                'episode': {
                    'r': torch.tensor([self.ep_reward]),
                    '_r':torch.tensor([True])
                },
            }
            info['_final_info'] = torch.tensor([True])
            info['final_observation'] = obs

        return obs, reward, done, done, info

    def sounder_ (self, x, y, thdot):
        abs_src_vel = np.abs(thdot * 1)  # v = w . r
        # compute ccw perpendicular vector. if angular velocity is
        # negative, we reverse it. then multiply by absolute velocity
        src_vel = np.array([-y, x])
        src_vel = (src_vel / np.linalg.norm(src_vel)) * np.sign(thdot) * abs_src_vel
        src_pos = np.array([x, y])

        frequencies = [
            modified_doppler_effect(
                self.original_frequency,
                obs_pos=rec.pos,
                obs_vel=np.zeros(2),
                obs_speed=0.0,
                src_pos=src_pos,
                src_vel=src_vel,
                src_speed=np.linalg.norm(src_vel),
                sound_vel=self.sound_vel) for rec in self.sound_receivers
        ]
        amplitudes = [
            inverse_square_law_observer_receiver(
                obs_pos=rec.pos, src_pos=src_pos)
            for rec in self.sound_receivers
        ]
        return np.array(list(zip(frequencies, amplitudes)))


    def render_(self, theta, mode='human', sound_channel=0, sound_duration=.1):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(theta + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(theta + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(10)
            pygame.display.flip()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def reset_mm(self, seed=0, num_initial_steps=1):
        try:
            self.seed(seed)
        except:
            Warning('Seeding failed')
        self.reset()
        self.ep_step = 0
        self.ep_reward = 0.

        if self._debug:
            self._debug_data = {
                'pos': [],
                'vel': [],
                'sound_receivers': [rec.pos for rec in self.sound_receivers],
                'sound': []
            }

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
            obs, _, _, _, info = self.step_mm(np.array([0.0]))

        return obs, info

    def close(self, out=None):
        super().close()

        if out:
            with open(out, 'wb') as filehandle:
                pickle.dump(
                    self._debug_data,
                    filehandle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    def get_state(self):
        return torch.from_numpy(self._get_obs()).unsqueeze(0)

    def project(self, state):

        if torch.is_tensor(state):
            state = state.numpy().reshape(-1)
        x, y, thdot = state
        theta = np.arctan2(y, x)



        snd_observation = self.sounder_(x, y, thdot)
        snd_observation[:, 0] = snd_observation[:, 0] / 600  # normalize freq

        img_observation = self.render_(theta, mode=self.rendering_mode)[100:400, 100:400, :]
        rgb = Image.fromarray(img_observation)
        img_observation = np.array(rgb.resize((100,100)))

        return dict(
            state=torch.tensor(state).unsqueeze(0),
            rgb=torch.from_numpy(img_observation).unsqueeze(0),
            sound=torch.from_numpy(snd_observation).unsqueeze(0))


def main():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import argparse

    parser = argparse.ArgumentParser(description='PongSound debugger')
    parser.add_argument(
        '--file', type=str, required=True, help='File with debug data')
    args = parser.parse_args()

    COLORS = ['forestgreen', 'cornflowerblue', 'darkorange', 'm']

    # Load data
    debug = pickle.load(open(args.file, 'rb'))
    positions = np.vstack(debug['pos'])
    velocities = np.vstack(debug['vel'])
    sounds = np.vstack(debug['sound'])

    sound_receiver_positions = debug['sound_receivers']
    sound_receiver_positions = np.vstack(sound_receiver_positions)
    n_sound_receivers = sound_receiver_positions.shape[0]

    # Plots
    fig, _ = plt.subplots()

    # - Plot ball data
    ax = plt.subplot('311')
    plt.xlim(LEFT_MARGIN, RIGHT_MARGIN)
    plt.ylim(BOTTOM_MARGIN, TOP_MARGIN)

    # -- Plot ball position
    plt.scatter(positions[:, 1], positions[:, 0], s=3, c='k')
    ball_plot, = plt.plot(positions[0, 1], positions[0, 0], marker='o')

    # -- Plot ball velocity
    vel_arrow = plt.arrow(
        positions[0, 1],
        positions[0, 0],
        velocities[0, 1],
        velocities[0, 0],
        width=4e-2)

    # -- Plot ball to mic line
    src_mic_plots = []
    for sr in range(n_sound_receivers):
        p, = plt.plot([positions[0, 1], sound_receiver_positions[sr, 1]],
                      [positions[0, 0], sound_receiver_positions[sr, 0]],
                      c=COLORS[sr])
        src_mic_plots.append(p)

    time_slider = Slider(
        plt.axes([0.2, 0.05, 0.65, 0.03]),
        'timestep',
        0,
        len(debug['pos']) - 1,
        valinit=0,
        valstep=1)

    # - Plot sound data
    plt.subplot('312')
    sound_marker_plots = []
    for sr in range(n_sound_receivers):
        plt.plot(sounds[:, sr], c=COLORS[sr])
        p, = plt.plot(1, sounds[1, sr], marker='o')
        sound_marker_plots.append(p)

    plt.subplot('313')
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    plt.plot(speeds)
    speed_marker_plot, = plt.plot(0, speeds[0], marker='o')

    def update(_):
        nonlocal vel_arrow
        timestep = int(time_slider.val)

        ball_position = debug['pos'][timestep]
        ball_plot.set_data(ball_position[1], ball_position[0])

        for sr in range(n_sound_receivers):
            src_mic_plots[sr].set_data(
                [positions[timestep, 1], sound_receiver_positions[sr, 1]],
                [positions[timestep, 0], sound_receiver_positions[sr, 0]])

        vel_arrow.remove()
        vel_arrow = ax.arrow(
            positions[timestep, 1],
            positions[timestep, 0],
            velocities[timestep, 1],
            velocities[timestep, 0],
            width=4e-2)
        for sr in range(n_sound_receivers):
            sound_marker_plots[sr].set_data(timestep, sounds[timestep, sr])

        speed_marker_plot.set_data(timestep, speeds[timestep])

        fig.canvas.draw_idle()

    def arrow_key_image_control(event):
        if event.key == 'left':
            time_slider.set_val(max(time_slider.val - 1, time_slider.valmin))
        elif event.key == 'right':
            time_slider.set_val(min(time_slider.val + 1, time_slider.valmax))

        update(0)

    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    time_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
