from skimage.util import random_noise
from skimage.color import gray2rgb
from Z_optimizer import generatePerturbation
import numpy as np
import cv2


img = np.array(cv2.imread("data/test_image.jpg"))
print(img.shape)
noise = generatePerturbation(img.shape[:-1], mean=0, var=0.1)
noise = gray2rgb(noise)
# gauss = noise.reshape(*img.shape)
img_gauss = cv2.add(img,noise)

# cv2.imshow('sample image', img_gauss)
#
# cv2.waitKey(0)  # waits until a key is pressed
# cv2.destroyAllWindows()  # destroys the window showing image

# Display the image
#
