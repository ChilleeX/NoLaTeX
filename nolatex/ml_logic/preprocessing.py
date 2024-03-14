import os
import cv2  #pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import keras
from skimage.exposure import is_low_contrast
from tensorflow.image import rgb_to_grayscale
from tensorflow.image import resize_with_pad

# DIRECTORY
#Who is running needs to change the path. 
image_dir = os.path.join('..','initial_test_data','batch_1_salmple10') #Dir containing images
archives = os.listdir(image_dir) #listing all dir files

#exit_dir = '/home/arturmarcon/code/ChilleeX/NoLateX/Binarization/binarization_img_otsu' 
#if not os.path.exists(exit_dir):
    #os.makedirs(exit_dir)

# FUNCTIONS
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
    
    # 3 Binarization image
    image_binary = convert_to_binary(resized_image)
    
    return image_binary


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


def convert_to_binary(image): #Function to binarizate one image
    _, binarization_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #applied the binarization
    return binarization_img

# Testing
# for archive in archives:
#     if archive.endswith('.jpg'):
#         image_path = os.path.join(image_dir, archive) 
#         image_raw = cv2.imread(image_path, 0) #load image
#         image_greyscale = image_preprocessing(image_raw, 254, 254)
#         image_lowContrast = is_low_contrast(image_raw, faction_threshold=.09) 
#         image_binary = convert_to_binary(image_lowContrast) #binarize image

#        filename = os.path.join(exit_dir, f'binarizada_{archive[:-4]}.jpg') #save binarized image
        
#        cv2.imwrite(filename, image_binary)

#        if os.path.exists(filename):
#            print(f'{archive} binarized image_{archive}')
#        else:
#            print(f'Erro: didnt save {archive}.')