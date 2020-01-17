# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import os
import pickle
import random, json
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers, utils
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ProgbarLogger, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.layers import Input

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = tf.shape(x)[-1]
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.acos(tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)

        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer,
        })
        return config

class CosFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = tf.shape(x)[-1]
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        target_logits = logits - self.m

        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer,
        })
        return config


class FineTuning():
    def __init__(self):
        '''
        学習済みモデルのロード(base_model)
        '''
        self.base_model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet')

    def createModel(self, label_dict):
        '''
        転移学習用のレイヤーを追加
        '''
        # y = Input(shape=(len(label_dict),))
        added_layer = GlobalAveragePooling2D()(self.base_model.output)
        added_layer = Dense(len(label_dict), activation='softmax', name='classification')(added_layer)
        # added_layer = ArcFace(len(label_dict), regularizer=regularizers.l2(1e-4))([added_layer, y])

        '''
        base_modelと転移学習用レイヤーを結合
        '''
        # model = Model(inputs=[self.base_model.input, y], outputs=added_layer)
        model = Model(inputs=self.base_model.input, outputs=added_layer)

        '''
        base_modelのモデルパラメタは学習させない。
        (added_layerのモデルパラメタだけを学習させる)
        '''
        ''' xception '''
        for layer in self.base_model.layers[:108]:
            layer.trainable = False
            if layer.name.startswith('batch_normalization'):
                layer.trainable = True
            if layer.name.endswith('bn'):
                layer.trainable = True
        for layer in self.base_model.layers[108:]:
            layer.trainable = True

        model.summary()
        return model

class DataSequence(Sequence):
    def __init__(self, data_path, label, batch_size, is_valid=False):
        self.batch = batch_size
        self.data_file_path = data_path
        self.datagen = ImageDataGenerator(
                            rotation_range=15,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.1,
                            # horizontal_flip=True,
                            # channel_shift_range=3.,
                            # brightness_range=[0.95, 1.05]
                        )
        self.is_valid = is_valid
        d_list = os.listdir(self.data_file_path)
        self.f_list = []
        for dir in d_list:
            if dir == 'empty': continue
            for f in os.listdir(self.data_file_path+'/'+dir):
                self.f_list.append(self.data_file_path+'/'+dir+'/'+f)
        self.label = label
        self.length = len(self.f_list)

    def __getitem__(self, idx):
        warp = self.batch
        aug_time = 8 if not self.is_valid else 0
        datas, labels = [], []
        label_dict = self.label

        for f in random.sample(self.f_list, warp):
            img = cv2.imread(f)
            img = cv2.resize(img, (160, 160))
            img = img.astype(np.float32) / 255.0
            datas.append(img)
            label = f.split('/')[-2].split('_')[-1]
            labels.append(label_dict[label])
            # Augmentation image
            for num in range(aug_time):
                tmp = self.datagen.random_transform(img)
                datas.append(tmp)
                labels.append(label_dict[label])

        datas = np.asarray(datas)
        labels = pd.DataFrame(labels)
        labels = utils.to_categorical(labels, len(label_dict))
        # return [datas, labels], labels
        return datas, labels

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        ''' 何もしない'''
        pass

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=50):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
      
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

if __name__=="__main__":
    base_path = './Extract'

    label_dict = {}
    count = 0
    batch_size = 4
    for d_name in os.listdir(base_path):
        if d_name == 'empty': continue
        if d_name == '.DS_Store': continue
        d_name = d_name.split('_')[-1]
        label_dict[d_name] = count
        count += 1
    print(label_dict)
    pickle.dump(label_dict, open('./model/polaris_labels.pkl','wb'))
    train_gen = DataSequence(base_path, label_dict, batch_size)
    validate_gen = DataSequence(base_path, label_dict, batch_size)

    # モデル構築
    ft = FineTuning()
    model = ft.createModel(label_dict)
    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print(tf.test.gpu_device_name())

    # fit model
    model.fit_generator(
        train_gen,
        epochs=10,
        steps_per_epoch=int(train_gen.length / batch_size),
        # callbacks=callbacks,
        validation_data=validate_gen,
        validation_steps=int(validate_gen.length / batch_size),
    )

    # save model
    model.save('./model/polaris_facenet_model.h5')
