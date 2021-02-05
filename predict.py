# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:04:24 2021

@author: lnln2
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os

keras = tf.keras
layers = tf.keras.layers

covn_base = keras.applications.xception.Xception(weights = 'imagenet',
                                                 include_top = False,
                                                 input_shape=(256,256,3),
                                                 pooling='avg')

model = keras.Sequential()
model.add(covn_base)
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(7,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.load_weights('Xcp_weights.h5')


img_path = 'C:/Users/Matebook/Desktop/示例/2-A. flavus/1 (286).jpg'


image = tf.io.read_file(img_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [360, 360])
image = tf.image.random_crop(image, [256, 256, 3])
image = tf.cast(image, tf.float32)
image = image/255

pre_list = []
pre_list.append(image)
pre_list = np.array(pre_list)


classes = model.predict_classes(pre_list)
proba = model.predict_proba(pre_list)


print(proba)
print(classes)

