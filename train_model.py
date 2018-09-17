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

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ProgbarLogger, ReduceLROnPlateau, LambdaCallback

from keras.utils import np_utils, Sequence
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

class CNN:
    '''
    CNNで学習を行う。
    '''
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.shape = (128, 128, 3)

    '''
    Networkを定義する
    '''
    def createNetwork(self):
        model = Sequential()
        model.add(Conv2D(124, (5, 5), activation='relu',
            input_shape=self.shape))
        model.add(Conv2D(122, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='softmax'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))

        return model


class DataSequence(Sequence):
    def __init__(self, kind, length, data_path):
        self.kind = kind
        self.length = length
        self.data_file_path = data_path
        self.datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.5)
        d_list = os.listdir(self.data_file_path)
        self.f_list = []
        for dir in d_list:
            for f in os.listdir(self.data_file_path+'/'+dir):
                self.f_list.append(self.data_file_path+'/'+dir+'/'+f)

    def __getitem__(self, idx):
        warp = 20
        aug_time = 2
        datas, labels = [], []
        label_dict = {
            'FATE': 0, 'HA': 1, 'HAGAREN': 2, 'MADOMAGI': 3, 'SAO': 4, 'TOARU': 5
        }

        for f in random.sample(self.f_list, warp):
            img = cv2.imread(f)
            img = cv2.resize(img, (128, 128))
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
        labels = np_utils.to_categorical(labels, 6)
        return datas, labels

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        ''' 何もしない'''
        pass

if __name__=="__main__":
    base_path = './Images'
    d_list = os.listdir(base_path)
    print(d_list)

    model_file_name = "cnn.h5"

    # モデル構築
    cnn = CNN(6)
    model = cnn.createNetwork()
    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5', verbose=1, save_weights_only=True, monitor='val_loss'),
        # EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),
        ReduceLROnPlateau(factor=0.02, patience=1, verbose=1, cooldown=5, min_lr=1e-10),
        LambdaCallback(on_batch_begin=lambda batch, logs: print(' now: ',   datetime.datetime.now()))
    ]

    # fit model
    step_size = 20
    file_all = 180
    train_gen = DataSequence('train', file_all, base_path)
    validate_gen = DataSequence('validate', file_all, base_path)
    model.fit_generator(
        train_gen,
        steps_per_epoch=3*int(file_all/step_size),
        epochs=300,
        validation_data=validate_gen,
        validation_steps=int(file_all/step_size),
        callbacks=callbacks
        )

    # save model
    model.save(model_file_name)