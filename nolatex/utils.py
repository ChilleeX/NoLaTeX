import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
from preprocessing import image_preprocessing


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


def resample_json(img_path, json_path):
  """takes a path to a folder of images and the path to the JSON file
  returns a dataframe with only the rows corresponding to the pictures in the file"""
  dir_files = os.listdir(img_path)
  uuids = [file.replace('.jpg','') for file in dir_files]
  matches = info["uuid"].isin(uuids)
  indices = np.where(matches)[0]
  index_list = indices.tolist()
  json = pd.read_json(json_path)
  json_resampled = json[json.index.isin(index_list)]
  json_resampled.reset_index(drop=True, inplace=True)
  return json_resampled

def load_data_to_dict(
        json_path: str,
        img_dir:str,
    ) -> Dict:

    # Loading the JSON file

    data = resample_json(img_dir, json_path)

    # Creating a list of visible Latex characters
    latex_chars = [data['image_data'][index]['visible_latex_chars'] for index in range(len(data['image_data']))]

    # # Creating Char Dict
    # class_dict = {}
    # unique_chars = []
    # for chars in range(len(latex_chars)):
    #     for char in latex_chars[chars]:
    #         unique_chars.append(char)
    #         unique_chars = list(set(unique_chars))

    # dict_keys = np.arange(0, len(unique_chars))
    # for key in dict_keys:
    #     for char in unique_chars:
    #         class_dict[key] = char

     #extracting classes
    classes = [data['image_data'][index]['visible_latex_chars'] for index in range(len(data['image_data']))]
    #Class_ids contais the unique classes
    class_ids = list(set([ele for sublist in classes for ele in sublist]))
    #mapping the classes
    mapping = {string: _ for _, string in enumerate(class_ids)}
    #converting the classes to numbers
    classes = [list(map(mapping.get, ele)) for ele in classes]
    classes = np.array(classes)
    #defining the right class mapping fo the model(inverse as mapping)
    class_mapping = dict(zip(range(len(class_ids)), class_ids))

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
    preprocessed_images = []
    for image in loadedImages:
        preprocessed_images.append(image_preprocessing(image))

    data_dict = {
        "images": preprocessed_images,
        "bounding_boxes": {
            "classes": np.expand_dims(np.arange(0,len(latex_chars[5])), axis=0),
            "boxes": np.expand_dims(bboxs, axis=0)
            }
        }

    return data_dict, class_mapping
