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
from keras.preprocessing.image import ImageDataGenerator

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
        elif base_model == 'DenseNet121':
            self.base = DenseNet121(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
        else:
            self.base = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
            # self.base = ResNet50(include_top=False, weights='imagenet', input_tensor=self.input_tensor)

    def getOptimizer(self):
        if self.base_model == 'VGG16':
            # opt = SGD(lr=1e-6, momentum=0.9)
            opt = Adam(lr=1e-4)
        elif self.base_model == 'DenseNet121':
            opt = SGD(lr=1e-6, momentum=0.9)
            # opt = Adam(lr=1e-4)
        else:
            # opt = SGD(lr=1e-6)
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
        print(len(model.layers))
        for layer in model.layers[:12]:
             layer.trainable = False
        # for layer in model.layers[:139]: # default 179
        #     if 'BatchNormalization' not in str(layer):
        #         layer.trainable = False
        return model

class DataSequence(Sequence):
    def __init__(self, kind, length, data_path, label):
        self.kind = kind
        self.length = length
        self.data_file_path = data_path
        self.datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.5)
        d_list = os.listdir(self.data_file_path)
        self.f_list = []
        for dir in d_list:
            for f in os.listdir(self.data_file_path+'/'+dir):
                self.f_list.append(self.data_file_path+'/'+dir+'/'+f)
        self.label = label

    def __getitem__(self, idx):
        warp = 30
        aug_time = 2
        datas, labels = [], []
        label_dict = self.label

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
        labels = np_utils.to_categorical(labels, len(label_dict))
        return datas, labels

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        ''' 何もしない'''
        pass

if __name__=="__main__":
    model = sys.argv[1]

    base_path = './Images'
    d_list = os.listdir(base_path)
    print(d_list)
    label_dict = json.load(open('./model/category.json', 'r'))

    model_file_name = "funiture_cnn.h5"

    # モデル構築
    ft = FineTuning(len(label_dict), model)
    model = ft.createNetwork()
    opt = ft.getOptimizer()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint('./model/checkpoints/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5', verbose=1, save_weights_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),
        ReduceLROnPlateau(factor=0.02, patience=1, verbose=1, cooldown=5, min_lr=1e-10),
        LambdaCallback(on_batch_begin=lambda batch, logs: print(' now: ',   datetime.datetime.now()))
    ]

    # fit model
    # model.fit(datas, labels, batch_size=50, epochs=n_epoch, callbacks=callbacks, validation_split=0.1)
    step_size = 30
    file_all = 180
    train_gen = DataSequence('train', file_all, base_path, label_dict)
    validate_gen = DataSequence('validate', file_all, base_path, label_dict)
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
