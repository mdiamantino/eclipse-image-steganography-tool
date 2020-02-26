import imgaug as ia
import numpy as np

"""
p = Augmentor.Pipeline("/home/mdc/PycharmProjects/eclipse/data")
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.skew_tilt(probability=0.5, magnitude=0.7)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=0.7, percentage_area=0.5)
p.sample(1)
"""
from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data


ia.seed(1)
