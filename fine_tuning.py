# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import datetime
import random
import requests
import numpy as np
import pandas as pd
import configparser

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ProgbarLogger, ReduceLROnPlateau, LambdaCallback
from keras.layers import Input

from keras.utils import np_utils, Sequence
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201, DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator

class FineTuning:
    '''
    CNNで学習を行う。(転移学習)
    Avable base_model is
        VGG16, DenseNet201, ResNet50
    '''
    def __init__(self, config, num_classes):
        self.num_classes = num_classes
        self.shape = (
                    int(config['PARAM']['width']),
                    int(config['PARAM']['height']),
                    int(config['PARAM']['channel'])
                    )
        self.input_tensor = Input(shape=self.shape)
        self.base = VGG16(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
        # self.base = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=self.input_tensor)

    def getOptimizer(self):
        opt = Adam(lr=1e-4)
        return opt

    '''
    Networkを定義する
    '''
    def createNetwork(self):
        tmp_model = Sequential()
        tmp_model.add(Flatten(input_shape=self.base.output_shape[1:]))
        tmp_model.add(Dense(1024, activation='relu'))
        tmp_model.add(Dropout(0.25))
        tmp_model.add(Dense(512, activation='relu'))
        tmp_model.add(Dense(self.num_classes, activation='softmax'))

        model = Model(inputs=self.base.input, outputs=tmp_model(self.base.output))
        print(len(model.layers))
        for layer in self.base.layers[:-4]:
             layer.trainable = False
        return model


class DataSequence(Sequence):
    def __init__(self, config, kind, length, data_path, label):
        self.batch = int(config['PARAM']['batch'])
        self.kind = kind
        self.length = length
        self.data_file_path = data_path
        self.datagen = ImageDataGenerator(
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.3
                        )
        d_list = os.listdir(self.data_file_path)
        self.f_list = []
        for dir in d_list:
            for f in os.listdir(self.data_file_path+'/'+dir):
                self.f_list.append(self.data_file_path+'/'+dir+'/'+f)
        self.label = label

    def __getitem__(self, idx):
        warp = self.batch
        aug_time = 3
        datas, labels = [], []
        label_dict = self.label
        size = (int(config['PARAM']['width']), int(config['PARAM']['height']))

        for f in random.sample(self.f_list, warp):
            img = cv2.imread(f)
            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0
            datas.append(img)
            label = f.split('/')[2].split('_')[1]
            labels.append(label_dict[label])
            # Augmentation image
            for num in range(aug_time):
                tmp = self.datagen.random_transform(img)
                datas.append(tmp)
                labels.append(label_dict[label])

        datas = np.asarray(datas)
        labels = pd.DataFrame(labels)
        labels = np_utils.to_categorical(labels, len(label_dict))
        return datas, labels

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        ''' 何もしない'''
        pass

if __name__=="__main__":

    ''' 設定ファイルの読み込み '''
    config = configparser.ConfigParser()
    config.read('./model/config.ini')

    base_path = config['PATH']['img']
    d_list = os.listdir(base_path)
    print(d_list)
    label_dict = json.load(open(config['PATH']['category'], 'r'))

    model_file_name = "model/funiture_cnn.h5"

    # モデル構築
    ft = FineTuning(config, len(label_dict))
    model = ft.createNetwork()
    opt = ft.getOptimizer()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(config['PATH']['chkpnt'], verbose=1, save_weights_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),
        LambdaCallback(on_batch_begin=lambda batch, logs: print(' now: ',   datetime.datetime.now()))
    ]

    # fit model
    step_size = int(config['PARAM']['batch'])
    file_all = 1070
    train_gen = DataSequence(config, 'train', file_all, base_path, label_dict)
    validate_gen = DataSequence(config, 'validate', file_all, base_path, label_dict)
    model.fit_generator(
        train_gen,
        steps_per_epoch=4*int(file_all/step_size),
        epochs=300,
        validation_data=validate_gen,
        validation_steps=4*step_size,
        callbacks=callbacks
        )

    # save model
    model.save(model_file_name)
