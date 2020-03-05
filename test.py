import numpy as np
import cv2
from PIL import Image
image = cv2.imread("data/test_image.jpg")

row, col, ch = image.shape
mean = 0
var = 1
sigma = var ** 0.5
gauss = np.random.normal(mean, sigma, image.shape)
gauss = gauss.reshape(image.shape)
noisy = image + gauss
