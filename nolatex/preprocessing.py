import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import keras
from skimage.exposure import is_low_contrast
from tensorflow.image import rgb_to_grayscale
from tensorflow.image import resize_with_pad

def image_preprocessing(image, height, width):
    # 1 Convert RBG in Grayscale
    gray_image = rgb_to_grayscale(image)

    # 2 Resize image to 254x254 with padding
    resized_image = resize_with_pad(
        gray_image,
        target_height=height,
        target_width=width,
        method="area",
        antialias=False
    )

    return resized_image

def rm_lowContrast_img(imagelist, faction_threshold=.09):

    n_img = len(imagelist)

    imagelist_good = []
    imagelist_bad = []

    for i in range(n_img):
        if is_low_contrast(imagelist[i], fraction_threshold=faction_threshold):
            imagelist_bad.append(imagelist[i])
        else:
            imagelist_good.append(imagelist[i])

    return imagelist_good, imagelist_bad
