from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


from utils_noise import perlin

# Inception v3

# Specify image dimensions
length, height = 1920, 1200


def colorize(noise, color=[1, 1, 1]):
    """
    Color noise
    :param noise: Has dimension 2 or 3, pixel range [0, 255]
    :param color: is [a, b, c] where a, b, c are from {-1, 0, 1}
    :return:
    """
    if noise.ndim == 2:  # expand to include color channels
        noise = np.expand_dims(noise, 2)
    return (noise - 0.5) * color * 2  # output pixel range [-1, 1]


def perlinNoiseFunc(params, height, length):
    freq, freq_sine, octave = params
    noise = perlin(height, length, 1 / freq, int(octave), freq_sine)
    return colorize(noise)


def perturb(img, noise, norm):
    """
    Perturb image and clip to maximum perturbation norm
    :param img: image with pixel range [0, 1]
    :param noise: noise with pixel range [-1, 1]
    :param norm: L-infinity norm constraint
    :return: Perturbed image
    """
    noise = np.sign((noise - 0.5) * 2) * norm
    noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
    return (img + noise)


# Parameter boundaries for Bayesian optimization
bounds = [{"name": "freq", "type": "continuous", "domain": (1 / 160, 1 / 20),
           "dimensionality": 1},
          {"name": "freq_sine", "type": "continuous", "domain": (4, 32),
           "dimensionality": 1},
          {"name": "octave", "type": "discrete", "domain": (1, 2, 3, 4),
           "dimensionality": 1}]

## Initialization
# Get original image and index
image_path = "data/test_image.jpg"
import cv2
orig_img = cv2.imread(image_path)
# img = image.load_img(image_path, target_size=(height, length))
# orig_img = image.img_to_array(img)

## Constraints
max_norm = 8
freq, freq_sine, octave = 1 / 30, 2, 2
params = (freq, freq_sine, octave)
payload = perturb(orig_img, perlinNoiseFunc(params, height, length), max_norm)
cv2.imwrite("cover_image.png", payload)

# image.save_img("cover_image.png", payload)
