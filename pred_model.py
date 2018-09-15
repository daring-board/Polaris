# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import random
import requests
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential
from keras.utils import np_utils, Sequence


def join_fn(dat):
    return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

class FineTuning:
    '''
    CNNで学習を行う。(転移学習)
    Avable base_model is
        VGG16, DenseNet201, ResNet50
    '''
    def __init__(self, num_classes, base_model):
        self.num_classes = num_classes
        self.shape = (128, 128, 3)
        self.input_tensor = Input(shape=self.shape)
        self.base_model = base_model
        if base_model == 'VGG16':
            self.base = VGG16(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
        elif base_model == 'DenseNet201':
            self.base = DenseNet201(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
        else:
            self.base = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
            # self.base = ResNet50(include_top=False, weights='imagenet', input_tensor=self.input_tensor)

    def getOptimizer(self):
        if self.base_model == 'VGG16':
            # opt = SGD(lr=1e-4, momentum=0.9)
            opt = Adam(lr=1e-4)
        elif self.base_model == 'DenseNet201':
            opt = Adam(lr=1e-4)
        else:
            opt = Adam(lr=1e-4)
        return opt

    '''
    Networkを定義する
    '''
    def createNetwork(self):
        tmp_model = Sequential()
        tmp_model.add(Flatten(input_shape=self.base.output_shape[1:]))
        tmp_model.add(Dense(256, activation='relu'))
        tmp_model.add(Dropout(0.5))
        tmp_model.add(Dense(self.num_classes, activation='softmax'))

        model = Model(input=self.base.input, output=tmp_model(self.base.output))
        for layer in model.layers[:15]:
            layer.trainable = False
        return model


if __name__=="__main__":
    base_path = './test_imgs'
    d_list = os.listdir(base_path)
    print(d_list)

    datas, labels = [], []
    label_dict = {
        'FATE': 0, 'HA': 1, 'HAGAREN': 2, 'MADOMAGI': 3, 'SAO': 4, 'TOARU': 5
    }

    f_list = []
    for dir in d_list:
        f_list.append(base_path+'/'+dir)

    imgs = []
    for f in f_list:
        img = cv2.imread(f)
        imgs.append(img)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        datas.append(img)

    datas = np.asarray(datas)

    model_file_name = "funiture_cnn.h5"
    ft = FineTuning(6, 'VGG16')
    model = ft.createNetwork()
    model.load_weights('checkpoints/weights.49-0.33-0.89-0.23-0.93.hdf5')
    pred_class = model.predict(datas)

    for idx in range(len(imgs)):
        print(pred_class[idx])
        cv2.imshow('img_%d'%idx, imgs[idx])
        cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
