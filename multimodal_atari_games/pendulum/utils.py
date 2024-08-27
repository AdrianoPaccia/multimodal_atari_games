import numpy as np
from gym.envs.classic_control import rendering
import gymnasium as gym
#import gym
import matplotlib.pyplot as plt
import pygame.gfxdraw
import pendulum_env_new as ps


class PendulumProjector:
    def __init__(self):
        self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
        self.env.reset()
        self.original_frequency: 440.0
        self.sound_velocity: 20.0
        self.sound_receivers = [
            ps.SoundReceiver(ps.SoundReceiver.Location[ss])
            for ss in ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
        ]

    def render_sound(self, state):
        th, thdot = state
        x, y = np.cos(th), np.sin(th)

        abs_src_vel = np.abs(thdot * 1)
        src_vel = np.array([-y, x])
        src_vel = (
            src_vel / np.linalg.norm(src_vel)) * np.sign(thdot) * abs_src_vel
        src_pos = np.array([x, y])

        self._frequencies = [
            ps.modified_doppler_effect(
                self.original_frequency,
                obs_pos=rec.pos,
                obs_vel=np.zeros(2),
                obs_speed=0.0,
                src_pos=src_pos,
                src_vel=src_vel,
                src_speed=np.linalg.norm(src_vel),
                sound_vel=self.sound_vel) for rec in self.sound_receivers
        ]
        self._amplitudes = [
            ps.inverse_square_law_observer_receiver(
                obs_pos=rec.pos, src_pos=src_pos)
            for rec in self.sound_receivers
        ]
        return np.array(zip(self._frequencies, self._amplitudes))


    def render_image(self,state):
        '''
        given s state = (th, dth) returns the img and sounds associated

        '''
        screen_dim = 500

        screen = pygame.Surface((screen_dim, screen_dim))


        surf = pygame.Surface((screen_dim, screen_dim))
        surf.fill((255, 255, 255))

        bound = 2.2
        scale = screen_dim / (bound * 2)
        offset = screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        pygame.gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
        pygame.gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))

        pygame.gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        pygame.gfxdraw.filled_circle(
            surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        pygame.gfxdraw.aacircle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        pygame.gfxdraw.filled_circle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )


        # drawing axle
        pygame.gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        pygame.gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )


if __name__ == "__main__":
    env = PendulumProjector()

    for _ in range(10):
        th = [np.pi] #np.random.uniform(low=-np.pi, high=np.pi, size=(1,))
        dth = np.random.uniform(low=-8, high=8, size=(1,))
        obs = (th[0], dth[0])
        img = env.render(obs)[100:400,100:400]
        plt.imshow(img)
        plt.title(f'(th, dth) = {th[0]:.2f}, {dth[0]:.2f}')
        plt.show()


