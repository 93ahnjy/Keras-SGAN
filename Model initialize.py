import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D,BatchNormalization, Dense, MaxPooling2D, Flatten, Dropout, Input, Activation
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os




class SimpleModel:

    def __init__(self, input_shape, class_num):

        self.input_shape = input_shape
        self.class_num = class_num

        self.train_data  = np.random.random((640, 32, 32, 3))    # 640 images, 32, 32, 3 shape image
        self.train_label = np.random.random((640, 90))           # 640 images, 90 classes,

        self.val_data = np.random.random((100, 32, 32, 3))
        self.val_labels = np.random.random((100, 90))



    def build_model(self):

        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape= self.input_shape, activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(8, (2, 2),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.class_num, activation='softmax'))

        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0011)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())
        return model



    def train(self):

        model = self.build_model()
        model.fit(self.train_data, self.train_label, epochs=10, batch_size=32)


#a = SimpleModel(input_shape=(32, 32, 3), class_num=90)
#a.train()










class MyLayer(Layer):
  def __init__(self, num_classes):
    super(MyLayer, self).__init__()
    self.num_outputs = num_classes

  def build(self, input_shape):
    self.kernel = self.add_variable(name="kernel",
                                    shape=[int(input_shape[-1]), self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

  # HOW TO USE?
  # inputs = tf.keras.Input(shape=(10,5))
  # outputs = MyLayer(20)(inputs)




class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        # 층을 정의합니다.
        self.conv5x5   = Conv2D(32, (5, 5), activation='relu', padding='valid')
        self.maxpool_1 = MaxPooling2D(pool_size=(2, 2))
        self.conv3x3   = Conv2D(16, (3, 3), activation='relu')
        self.maxpool_2 = MaxPooling2D(pool_size=(2, 2))
        self.conv2x2   = Conv2D(8, (2, 2), activation='relu')
        self.maxpool_3 = MaxPooling2D(pool_size=(2, 2))
        self.dropout   = Dropout(0.2)
        self.flatten   = Flatten()
        self.dense_1   = Dense(128, activation='relu')

        #self.dense_2   = Dense(self.num_classes, activation='softmax')
        self.custom_layer = MyLayer(num_classes=10)


    def call(self, inputs):

        # 정방향 패스를 정의합니다.
        # `__init__` 메서드에서 정의한 층을 사용합니다.
        x    = self.conv5x5(inputs)
        x    = self.maxpool_1(x)
        x    = self.conv3x3(x)
        x    = self.maxpool_2(x)
        x    = self.conv2x2(x)
        x    = self.maxpool_3(x)
        x    = self.dropout(x)
        x    = self.flatten(x)
        x    = self.dense_1(x)
        pred    = self.custom_layer(x)
        return pred

