import sys
import os
import numpy as np
import pandas as pd
import cv2, json
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import load_model, Model

import tensorflow as tf
from tensorflow.python.framework import ops

from fine_tuning import FineTuning

# Define model here ---------------------------------------------------
def build_model():
    """Function returning keras model instance.

    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    label_dict = json.load(open('./model/category.json', 'r'))

    # モデル構築
    ft = FineTuning(len(label_dict), 'VGG16')
    model = ft.createNetwork()
    model.load_weights('./model/checkpoints/weights.21-0.02-0.99-0.01-1.00.hdf5')
    model.summary()

    tmp = model.layers[-1]
    model1 = Model(inputs=model.input, outputs=model.layers[-2].output)
    model2 = Model(inputs=tmp.layers[0].input, outputs=tmp.layers[-2].output)
    return [model1, model2]

if __name__=="__main__":
    base_path = './Images'
    d_list = os.listdir(base_path)

    models = build_model()

    for d_name in d_list:
        imgs, paths = [], []
        d_path = base_path + '/' + d_name
        print(d_path)
        for f_name in os.listdir(d_path):
            if f_name == 'empty': continue
            f = d_path + '/' + f_name
            img = cv2.imread(f)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            imgs.append(img)
            paths.append(f)

        datas = np.asarray(imgs)
        vec = models[0].predict(datas)
        features = models[1].predict(vec)

        df = pd.DataFrame(features)
        df['path'] = pd.DataFrame(paths)
        df.to_csv('generate/features.csv', mode='a', header=False, index=False)
