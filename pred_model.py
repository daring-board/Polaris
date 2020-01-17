# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json, pickle
import random
import requests
import numpy as np
import pandas as pd
import configparser

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model

from fine_tuning import FineTuning

def cos_sim(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    norm = 1 if norm == 0 else norm
    return np.dot(v1, v2) / norm

if __name__=="__main__":
    ''' 設定ファイルの読み込み '''
    config = configparser.ConfigParser()
    config.read('./model/config.ini')

    base_path = './Extract/'
    d_list = os.listdir(base_path)
    print(d_list)

    label_dict = pickle.load(open('./model/polaris_labels.pkl','rb'))
    label_dict = {v: k for k, v in label_dict.items()}
    print(label_dict)

    f_list = []
    for d in d_list:
        for f in os.listdir(base_path+d):
            f_list.append(base_path+d+'/'+f)

    model_file_name = './model/polaris_facenet_model.h5'
    model = tf.keras.models.load_model(model_file_name, compile=False)
    
    for f in f_list:
        print(f)
        img = cv2.imread(f)                
        img = cv2.resize(img, (int(config['PARAM']['width']), int(config['PARAM']['height'])))
        img = img.astype(np.float32) / 255.0
        predict = model.predict(img[None])
        num = np.argmax(predict[0])
        print(label_dict[num])
