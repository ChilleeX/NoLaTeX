import os
import numpy as np
from cv2 import threshold, imread, THRESH_BINARY, THRESH_OTSU
from skimage.exposure import is_low_contrast

# FUNCTIONS
def image_preprocessing(image_path: str) -> np.array:
    # 1 Convert RBG in Grayscale
    gray_image = imread(image_path, 0)

    # 3 Binarization image
    image_binary = threshold(gray_image, 0, 255, THRESH_BINARY + THRESH_OTSU)

    preprocessed_img = np.expand_dims(image_binary[1], axis=-1)
    preprocessed_img = np.dstack((preprocessed_img, preprocessed_img))
    preprocessed_img = np.dstack((preprocessed_img, preprocessed_img))

    # #Creating two dummy channel with zeros
    # dummy_channels = np.ones((preprocessed_img.shape[0],preprocessed_img.shape[1],2),dtype=int)
    # #Stacking dummy channels into the image
    # preprocessed_img = np.dstack((preprocessed_img, dummy_channels))

    return preprocessed_img

def is_low_contrast_from_path(image_path: str, fraction_threshold: float) -> bool:
    image = imread(image_path)
    return is_low_contrast(image, fraction_threshold)


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
