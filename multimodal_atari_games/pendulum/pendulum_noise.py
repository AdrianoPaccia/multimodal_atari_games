import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImageNoise:
    def __init__(self, noise_types: list, config: dict):
        self.noise_types = noise_types
        self.config = config

    def apply_noise(self, noise_type:str, image):
        img = image.copy()
        params = self.config.get(noise_type, {})
        if noise_type == 'gaussian_noise':
            noisy_image = self.apply_gaussian_noise(img)
        elif noise_type == 'salt_pepper_noise':
            noisy_image = self.apply_salt_pepper_noise(img)
        elif noise_type == 'poisson_noise':
            noisy_image = self.apply_poisson_noise(img)
        elif noise_type == 'speckle_noise':
            noisy_image = self.apply_speckle_noise(img)
        elif noise_type == 'uniform_noise':
            noisy_image = self.apply_uniform_noise(img)
        elif noise_type == 'gaussian_blur':
            noisy_image = self.apply_gaussian_blur(img)
        elif noise_type == 'motion_blur':
            noisy_image = self.apply_gaussian_blur(img)
        elif noise_type == 'quantization_noise':
            noisy_image = self.apply_quantization_noise(img)
        elif noise_type == 'confounders_noise':
            noisy_image = self.apply_confounders_noise(img)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return noisy_image

    def apply_random_noise(self, image):
        noise_type = random.choice(list(self.config.keys()))
        return self.apply_noise(noise_type, image), noise_type

    def apply_all_noises(self,image):
        noisy_images = []
        for noise_type in self.noise_types:
            noisy_images.append(self.apply_noise(noise_type, image))
        return noisy_images, self.noise_types
    
    #all implemented noises

    def apply_gaussian_noise(self, image):
        mean = self.config['gaussian_noise']['mu']
        stddev = self.config['gaussian_noise']['std']
        noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def apply_salt_pepper_noise(self, image):
        ratio = self.config['salt_pepper_noise']['ratio']
        salt_pepper = np.random.rand(*image.shape)
        noisy_image = np.copy(image)
        noisy_image[salt_pepper < ratio] = 0
        noisy_image[salt_pepper > 1 - ratio] = 255
        return noisy_image

    def apply_poisson_noise(self, image):
        noisy_image = np.random.poisson(image)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
        #lam = self.config['poisson_noise']['lam']
        #noisy_image = np.random.poisson(lam, image.shape)
        #return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def apply_speckle_noise(self, image):
        noise = np.random.normal(0, 1, size=image.shape)
        mean = self.config['speckle_noise']['mean']
        std = self.config['speckle_noise']['std']
        noisy_image = np.random.normal(mean, std, image.shape) * image + image
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def apply_uniform_noise(self, image):
        min_val = self.config['uniform_noise']['min_val']
        max_val = self.config['uniform_noise']['max_val']
        noise = np.random.uniform(min_val, max_val, size=image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def apply_gaussian_blur(self, image):
        sigma = self.config['gaussian_blur']['sigma']
        noisy_image = gaussian_filter(image, sigma=sigma)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def apply_motion_blur(self, image):
        kernel_size = self.config['motion_blur']['kernel_size']
        angle = self.config['motion_blur']['angle']
        kernel = self._motion_blur_kernel(kernel_size, angle)
        noisy_image = convolve2d(image, kernel, mode='same', boundary='wrap')
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def apply_quantization_noise(self, image):
        bits = self.config['quantization_noise']['bits']
        noise = np.random.uniform(0, 1, size=image.shape)
        levels = 2 ** bits - 1
        noisy_image = (image / 255 * levels) / levels * 255
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def apply_confounders_noise(self, image):
        noisy_image = image.copy()
        max_patch_size = self.config['confounders_noise']['max_size']
        min_patch_size = self.config['confounders_noise']['min_size']
        max_num_patches = self.config['confounders_noise']['max_num_patches']
        n = random.randint(1, max_num_patches)
        height, width, _ = image.shape

        for _ in range(n):
            side_1 = random.randint(min_patch_size, max_patch_size)
            side_2 = random.randint(min_patch_size, max_patch_size)
            x = random.randint(0, width - side_1)
            y = random.randint(0, height - side_2)
            print(np.array(noisy_image).shape)
            for i in range(3):
                noisy_image[y:y+side_1, x:x+side_2,i] = random.uniform(0,225)
        return noisy_image

    def _motion_blur_kernel(self, size, angle):
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.sin(np.deg2rad(angle))
        kernel[int((size-1)/2), int((size-1)/2)] = np.cos(np.deg2rad(angle))
        kernel /= np.sum(kernel)
        return kernel

    def render_images(self, image_list, noise_types):
        n = len(image_list)
        fig, axes = plt.subplots(1, n , figsize=(10, 5))  # Create a figure with two subplots
        for i in range(n):
            axes[i].imshow(image_list[i])  
            axes[i].axis('off')  
            axes[i].set_title(f'{noise_types[i]}')  
        plt.show()



class SoundNoise:
    def __init__(self, noise_types: list, config: dict):
        self.noise_types = noise_types
        self.config = config

    def apply_noise(self, noise_type:str, sound):
        noisy_sound = sound.copy()
        params = self.config.get(noise_type, {})
        if noise_type == 'gaussian_noise':
            noisy_sound = self.apply_gaussian_noise(noisy_sound)
        elif noise_type == 'white_noise':
            noisy_sound = self.apply_white_noise(noisy_sound)
        elif noise_type == 'pink_noise':
            noisy_sound = self.apply_pink_noise(noisy_sound)
        elif noise_type == 'brownian_noise':
            noisy_sound = self.apply_brownian_noise(noisy_sound)
        elif noise_type == 'blue_noise':
            noisy_sound = self.apply_blue_noise(noisy_sound)
        elif noise_type == 'poisson_noise':
            noisy_sound = self.apply_poisson_noise(noisy_sound)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return noisy_sound

    def apply_random_noise(self, sound):
        noise_type = random.choice(list(self.config.keys()))
        return self.apply_noise(noise_type, sound), noise_type

    def apply_all_noises(self,sound):
        noisy_sounds = []
        for noise_type in self.noise_types:
            noisy_sounds.append(self.apply_noise(noise_type, sound))
        return noisy_sounds, self.noise_types

    #all implemented noises
    def apply_gaussian_noise(self, sound):
        mean = self.config['gaussian_noise']['mu']
        std = self.config['gaussian_noise']['std']
        noise = np.random.normal(mean, std, len(sound))
        noisy_sound = sound + noise
        return noisy_sound

    def apply_white_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        noisy_sound = sound + noise
        return noisy_sound


    def apply_pink_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        pink_noise = np.convolve(noise, np.ones(10)/10, mode='same')
        noisy_sound = sound + pink_noise
        return noisy_sound

    def apply_brownian_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        brownian_noise = np.cumsum(noise)
        noisy_sound = sound + brownian_noise
        return noisy_sound

    def apply_blue_noise(self, sound):
        noise = np.random.normal(0, 1, len(sound))
        blue_noise = np.cumsum(noise)
        noisy_sound = sound + blue_noise
        return noisy_sound

    def apply_poisson_noise(self, sound):
        mean = self.config['poisson_noise']['mean']
        poisson_noise = np.random.poisson(mean, len(sound))
        noisy_sound = sound + poisson_noise
        return noisy_sound

    def print_sounds(self,sounds,noise_types):
        n = len(sounds)
        for sound, noise in zip(sounds,noise_types):
            print(f'noise: {[round(x,3) for x in sound]}')
        
