import sys
import os
import numpy as np
import pandas as pd
import cv2, json
import configparser

from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import load_model, Model
import tensorflow as tf
from tensorflow.python.framework import ops

from fine_tuning import FineTuning

# Define model here ---------------------------------------------------
def build_model(config):
    label_dict = json.load(open(config['PATH']['category'], 'r'))

    # モデル構築
    ft = FineTuning(config, len(label_dict))
    model = ft.createNetwork()
    model.load_weights(config['PATH']['use_chkpnt'])
    model.summary()

    tmp = model.layers[-1]
    model1 = Model(inputs=model.input, outputs=model.layers[-2].output)
    model2 = Model(inputs=tmp.layers[0].input, outputs=tmp.layers[-2].output)
    return [model1, model2]

if __name__=="__main__":
    ''' 設定ファイルの読み込み '''
    config = configparser.ConfigParser()
    config.read('./model/config.ini')

    size = (int(config['PARAM']['width']), int(config['PARAM']['height']))

    base_path = config['PATH']['img']
    d_list = os.listdir(base_path)

    models = build_model(config)

    for d_name in d_list:
        imgs, paths = [], []
        d_path = base_path + '/' + d_name
        print(d_path)
        for f_name in os.listdir(d_path):
            if f_name == 'empty': continue
            f = d_path + '/' + f_name
            img = cv2.imread(f)
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0
            imgs.append(img)
            paths.append(f)

        datas = np.asarray(imgs)
        vec = models[0].predict(datas)
        features = models[1].predict(vec)

        df = pd.DataFrame(features)
        df['path'] = pd.DataFrame(paths)
        df.to_csv('generate/features.csv', mode='a', header=False, index=False)
