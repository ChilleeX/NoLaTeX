import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras_cv.layers
from tensorflow.keras.utils import image_dataset_from_directory
from keras.utils import load_imgfrom skimage.color import rgb2gray



def image_preprocessing(image):
    # 1 Convert RBG in Grayscale
    to_grayscale = keras_cv.layers.Grayscale(output_channels=1)
    gray_image = to_grayscale(image)

    # 2 Resize image to 254x254 with padding
    resize_img = keras_cv.layers.Resizing(
        254,
        254,
        interpolation="area",
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=True,
        bounding_box_format="xywh",
    )

    resized_img = resize_img(gray_image)
