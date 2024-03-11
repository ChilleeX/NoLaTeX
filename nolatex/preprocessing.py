import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras_cv.layers import Resizing
from keras_cv.layers import Grayscale
from keras.utils import image_dataset_from_directory
from keras.utils import load_img

def image_preprocessing(image, height, width):
    # 1 Convert RBG in Grayscale
    to_grayscale = Grayscale(output_channels=1)
    gray_image = to_grayscale(image)

    # 2 Resize image to 254x254 with padding
    resize_img = Resizing(
        height,
        width,
        interpolation="area",
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=True,
        bounding_box_format="xywh",
    )

    resized_img = resize_img(gray_image)

    return resized_img
