import os
import onnxruntime as rt
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt




def resize(img, w=256, h=256):
    p = max(img.shape[:2] / np.array([h, w]))
    s = img.shape[:2]
    r = s / p

    img = cv2.resize(img, (int(r[1]), int(r[0])))

    re = np.zeros((h, w, 3))
    offset = np.array((np.array(re.shape[:2]) - np.array(img.shape[:2])) / 2, dtype=np.int32)
    re[offset[0]:offset[0] + img.shape[0], offset[1]:offset[1] + img.shape[1]] = img
    return re

def load(image_file, w, h):
    i = cv2.imread(image_file)[...,::-1]
    input_image = resize(i, w, h) / 255
    return input_image[None].astype(np.float32)


def make_prediction(image_path):

    ltx_index = json.load((open('/home/diegoberan/code/ChilleeX/NoLaTeX/nolatex/ml_logic/keys.json')))
    providers = ['CPUExecutionProvider']
    model = rt.InferenceSession('/home/diegoberan/code/ChilleeX/NoLaTeX/nolatex/ml_logic/model.onnx', providers=providers)
    output_name = model.get_outputs()[0].name
    input_name = model.get_inputs()[0].name

    img = load(image_path, w=1024, h=192)
    res = model.run([output_name], {input_name: img})[0][0]
    res = np.argmax(res, axis=1)

    l = ''.join([ltx_index[str(x - 1)] if x != 0 else ' ' for x in res])

    return l
