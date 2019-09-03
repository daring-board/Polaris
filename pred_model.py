# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import random
import requests
import numpy as np
import pandas as pd
import configparser

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input
from keras.models import Sequential
from keras.utils import np_utils, Sequence

from fine_tuning import FineTuning

if __name__=="__main__":
    ''' 設定ファイルの読み込み '''
    config = configparser.ConfigParser()
    config.read('./model/config.ini')

    base_path = './Img_test'
    d_list = os.listdir(base_path)
    print(d_list)

    datas, labels = [], []
    label_dict = json.load(open(config['PATH']['category'], 'r'))
    print(label_dict)

    f_list = []
    for dir in d_list:
        f_list.append(base_path+'/'+dir)

    imgs = []
    for f in f_list:
        img = cv2.imread(f)
        imgs.append(img)
        img = cv2.resize(img, (int(config['PARAM']['width']), int(config['PARAM']['height'])))
        img = img.astype(np.float32) / 255.0
        datas.append(img)

    datas = np.asarray(datas)

    model_file_name = './model/models/polaris_custum_model.h5'
    ft = FineTuning(config, len(label_dict))
    model = ft.createNetwork()
    model.load_weights(model_file_name)
    # model = load_model('./model/models/polaris_custum_model.h5')
    pred_class = model.predict(datas)

    l_list = list(label_dict.keys())
    for idx in range(len(imgs)):
        for x in range(len(pred_class[idx])):
            print('%s, %f'%(l_list[x], pred_class[idx][x]))
        print()
        cv2.imshow('img_%d'%idx, imgs[idx])
        cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
