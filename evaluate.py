# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:01:56 2021

@author: lnln2
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os

keras = tf.keras
layers = tf.keras.layers

def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [360, 360])
    image = tf.image.random_crop(image, [256, 256, 3])
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    image = tf.cast(image, tf.float32)
    image = image/255
    label = tf.reshape(label, [1])
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 20

test_image_path = glob.glob('./dataset/ValidationSet/*/*.jpg')

test_image_label = [int(p.split('\\')[1].split('-')[0]) for p in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.batch(BATCH_SIZE)

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
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])


model.load_weights('Xception_weights.h5')

output = model.evaluate(test_image_ds,verbose=0)

print('test loss:',output[0])
print('accuracy:',output[1])





