# -*- coding: utf-8 -*-
"""
Convolutional Neural Network
Problem : To create a model that can predict if the image belongs to one class or the other
@author: paras
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()
model.add(Convolution2D(64, 2,2 , input_shape = (64, 64, 3) ,activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2), strides=(2,2)))
model.add(Convolution2D(128, 2,2  ,activation='relu'))
model.add(MaxPooling2D(pool_size= (2,2), strides=(2,2)))
model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 64, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

trainDataGen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
                             rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,)

testDataGen = ImageDataGenerator(rescale = 1./255)

train = trainDataGen.flow_from_directory('dataset/training_set',
                                         target_size = (64, 64), batch_size = 32, class_mode = 'binary')

test = testDataGen.flow_from_directory('dataset/test_set', target_size = (64, 64),
                                        batch_size = 32, class_mode = 'binary')

model.fit_generator(train, verbose=2, steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test,
                         validation_steps = 2000)