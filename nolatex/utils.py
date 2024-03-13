import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img


def loadImagesAsList(
        img_path:str
    ) -> List:
    # return array of images

    #get a list of file names
    imagesList = os.listdir(img_path)

    # Collect images from filename list
    loadedImages = []
    for image in imagesList:
      if 'Zone.Identifier' not in image:
        img = load_img(os.path.join(path, image))
        loadedImages.append(img)

    return loadedImages



def load_data_to_dict(
        json_path: str,
        img_dir:str,
    ) -> Dict:

    # Loading the JSON file
    data = pd.read_json(json_path)

    # Creating a list of visible Latex characters
    latex_chars = [data['image_data'][index]['visible_latex_chars'] for index in range(len(data['image_data']))]

    # Creating Char Dict
    class_dict = {}
    unique_chars = []
    for chars in range(len(latex_chars)):
        for char in latex_chars[chars]:
            unique_chars.append(char)
            unique_chars = list(set(unique_chars))

    dict_keys = np.arange(0, len(unique_chars))
    for key in dict_keys:
        for char in unique_chars:
            class_dict[key] = char


    # Creating list of Bounding Boxes
    bboxs = []
    for img_pos in range(len(data['image_data'])):
        X = data['image_data'][img_pos]['xmins_raw']
        Y = data['image_data'][img_pos]['ymins_raw']
        W = np.array(data['image_data'][img_pos]['xmaxs_raw']) - np.array(data['image_data'][img_pos]['xmins_raw'])
        H = np.array(data['image_data'][img_pos]['ymaxs_raw']) - np.array(data['image_data'][img_pos]['ymins_raw'])
        bbox = [[X[i],Y[i],W[i],H[i]] for i in range(len(X))]
        bboxs.append(bbox)

    # Creating list of image names
    img_filenames = list(data['filename'])

    # Collect images from filename list
    loadedImages = []
    for filename in img_filenames:
        img = load_img(os.path.join(img_dir, filename))
        loadedImages.append(img)



    data_dict = {
        "images": loadedImages,
        "bounding_boxes": {
            "classes": np.expand_dims(np.arange(0,len(latex_chars[5])), axis=0),
            "boxes": np.expand_dims(bboxs, axis=0)
            }
        }

    return data_dict
