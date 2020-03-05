import cv2
import numpy as np

image = cv2.imread("data/test_image.jpg")

row, col, ch = image.shape
mean = 0
var = 1
sigma = var ** 0.5

gaussian = np.random.normal(mean, sigma, image.shape[:2])
noisy = np.copy(image)
noisy[:, :, 0] = image[:, :, 0] + gaussian
noisy[:, :, 1] = image[:, :, 1] + gaussian
noisy[:, :, 2] = image[:, :, 2] + gaussian
