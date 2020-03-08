import cv2
import numpy as np

image = cv2.imread("data/test_image.jpg")

row, col, ch = image.shape
mean = 0
var = 1
sigma = var ** 0.5

new = np.random.normal(mean, sigma, (2, 3,3))

print(new)
print()
new[0]=new[1]
new[1]=np.zeros(new.shape[1:])
print(new)