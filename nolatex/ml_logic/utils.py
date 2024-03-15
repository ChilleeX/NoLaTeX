import numpy as np
import pandas as pd
import os
from cv2 import imread
from skimage.exposure import is_low_contrast
from tensorflow import ragged
from nolatex.ml_logic.preprocessing import image_preprocessing, is_low_contrast_from_path

def load_dataset(
    img_dir: str, json_path: str,
    low_contrast_threshold: float=.09,
    save_low_contrast_imgs: bool=True
    ) -> tuple[dict, dict, tuple[list, list]]:

    ##################################
    # 1 Load initial data from paths #
    ##################################

    # 1 Read the full json file from json_path to pd.Dataframe
    json = pd.read_json(json_path)

    # 2 Loading the image filename list from img_dir
    dir_files = os.listdir(img_dir)

    ################################
    # 2 Preprocess and load images #
    ################################

    # Creating the preprocessed image list
    preprocessed_imgs = []
    preprocessed_img_names = []
    low_contrast_imgs = []
    low_contrast_imgs_names = []

    for filename in dir_files:                                              # Interating over the filename list
        img_path = os.path.join(img_dir, filename)                          # Creating the image path

        if is_low_contrast_from_path(img_path, low_contrast_threshold):     # Check if the image is low-contrast
            if save_low_contrast_imgs:                                      # Check if low contrast images should be saved

                # Save low contrast images and image filenames in lists
                low_contrast_imgs_names.append(filename)
                image = imread(img_path)
                low_contrast_imgs.append(image)

        else:

            # Preprocess rest of images and save them and image filenames in lists
            preprocessed_img_names.append(filename)
            image = image_preprocessing(img_path)
            preprocessed_imgs.append(image)

    preprocessed_imgs = ragged.constant(preprocessed_imgs)
    #preprocessed_imgs = preprocessed_imgs.to_tensor()

    ##########################################
    # 3 Loading required data from full json #
    ##########################################

    uuids = [file.replace('.jpg','') for file in dir_files]
    matches = json["uuid"].isin(uuids)
    indices = np.where(matches)[0]
    index_list = indices.tolist()
    json_ds = json[json.index.isin(index_list)]
    json_ds.reset_index(drop=True, inplace=True)

    #####################################
    # 4 Loading required bounding boxes #
    #####################################

    bboxs = []
    for img_pos in range(len(json_ds['image_data'])):
        X = json_ds['image_data'][img_pos]['xmins_raw']
        Y = json_ds['image_data'][img_pos]['ymins_raw']
        W = np.array(json_ds['image_data'][img_pos]['xmaxs_raw']) - np.array(json_ds['image_data'][img_pos]['xmins_raw'])
        H = np.array(json_ds['image_data'][img_pos]['ymaxs_raw']) - np.array(json_ds['image_data'][img_pos]['ymins_raw'])
        bbox = [[X[i],Y[i],W[i],H[i]] for i in range(len(X))]
        bboxs.append(bbox)
    bboxs = ragged.constant(bboxs)

    #########################################
    # 5 Loading required Latex Code Targets #
    #########################################
    #extracting classes
    classes = [json_ds['image_data'][index]['visible_latex_chars'] for index in range(len(json_ds['image_data']))]
    #Class_ids contais the unique classes
    class_ids = list(set([ele for sublist in classes for ele in sublist]))
    #mapping the classes
    mapping = {string: _ for _, string in enumerate(class_ids)}
    #converting the classes to numbers
    classes = [list(map(mapping.get, ele)) for ele in classes]
    classes = ragged.constant(classes)
    #defining the right class mapping fo the model(inverse as mapping)
    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    ##################################
    # 6 Contructing final dictionary #
    ##################################

    final_dict = {"images": preprocessed_imgs,"bounding_boxes":{"classes": classes , "boxes": bboxs}}

    return final_dict, class_mapping, (low_contrast_imgs_names, low_contrast_imgs)


###########
# ARCHIVE #
###########

def loadImagesAsList(
        img_path:str
    ) -> list:
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
    ) -> dict:

    # Loading the JSON file

    data = resample_json(img_dir, json_path)

    # Creating a list of visible Latex characters
    # OLD latex_chars = [data['image_data'][index]['visible_latex_chars'] for index in range(len(data['image_data']))]

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

    #TODO might need to be a tensor
    # classes = np.array(classes)

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
    # TODO sortout low contrast images
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
