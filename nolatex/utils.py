import os
from tensorflow.keras.utils import load_img


def loadImagesAsList(path):
    # return array of images

    #get a list of file names
    imagesList = os.listdir(path)

    # Collect images from filename list
    loadedImages = []
    for image in imagesList:
      if 'Zone.Identifier' not in image:
        img = load_img(os.path.join(path, image))
        loadedImages.append(img)

    return loadedImages
