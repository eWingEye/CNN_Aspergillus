# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:57:44 2021

@author: lnln2
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os

keras = tf.keras
layers = tf.keras.layers

train_image_path = glob.glob('D:/model/7-YAG-35/train/*/*.jpg')
train_image_label = [int(p.split('\\')[1].split('-')[0]) for p in train_image_path]

def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [360, 360])
    image = tf.image.random_crop(image, [256, 256, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.cast(image, tf.float32)
    image = image/255
    label = tf.reshape(label, [1])
    return image, label

train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
BATCH_SIZE = 20
train_count = len(train_image_path)
train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)


test_image_path = glob.glob('D:/model/7-YAG-35/test/*/*.jpg')
test_image_label = [int(p.split('\\')[1].split('-')[0]) for p in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.batch(BATCH_SIZE)
test_count = len(test_image_path)


tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs',histogram_freq=1,update_freq=50)

covn_base = keras.applications.xception.Xception(weights = 'imagenet',
                                                include_top = False,
                                                input_shape=(256,256,3),
                                                 pooling='avg')
covn_base.trainable = True

model = keras.Sequential()
model.add(covn_base)
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(7,activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

initial_epochs = 8

history = model.fit(train_image_ds,
                   #steps_per_epoch = train_count//BATCH_SIZE,
                   epochs=initial_epochs,
                   validation_data=test_image_ds,
                   validation_steps=test_count//BATCH_SIZE,
                   callbacks=[tb_callback])


model.save_weights('Xcp_weights.h5')


