import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, Flatten, BatchNormalization, Dense, MaxPooling2D, Flatten, Dropout, Input, Activation, GlobalAveragePooling2D, UpSampling2D, concatenate, GaussianNoise
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
import pylab





def vgg16(input_size, num_classes, pretrained):

    weight_decay = 0.0005
    last_activation = 'softmax' if num_classes > 1 else 'sigmoid'



    if pretrained:
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
        print(pretrained_model.summary())

        model = Sequential()
        model.add(pretrained_model)

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(num_classes))
        model.add(Activation(last_activation))


    else:
        model = Sequential()


        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=(input_size[0], input_size[1], 3), kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation(last_activation))

    print(model.summary())
    return model











def Resnet50(input_size, num_classes):

    weight_decay = 0.0005
    last_activation = 'softmax' if num_classes > 1 else 'sigmoid'

    pretrained_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3), pooling='avg')


    last = pretrained_model.layers[-1].output
    x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(last)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    x = Dense(num_classes, activation=last_activation, kernel_regularizer=regularizers.l2(weight_decay))(x)


    model = Model(pretrained_model.input, x)

    print(model.summary())
    print(last_activation, '\n\n\n')

    return model






######################################################################################################


class PassThrough(Layer):

  def __init__(self, pass_feature):
    super(PassThrough, self).__init__()
    self.pass_feature = pass_feature


  def call(self, inputs):

    # passthrough layer - start
    f1 = self.pass_feature[:, ::2, ::2, :]
    f2 = self.pass_feature[:, 1::2, ::2, :]
    f3 = self.pass_feature[:, ::2, 1::2, :]
    f4 = self.pass_feature[:, 1::2, 1::2, :]
    return concatenate([f1, f2, f3, f4, inputs], axis=3)



def Encoder(input_shape, arch_type='D'):
    img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)(img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    pass_feature = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(pass_feature)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='same')(x)

    # passthrough layer - end
    x = PassThrough(pass_feature=pass_feature)(x)
    outputs = MaxPooling2D(pool_size=(2, 2))(x)

    if arch_type == 'D':
        outputs = Flatten()(outputs)
        outputs = Dense(1, activation='sigmoid')(outputs)

    Encoder = tf.keras.Model(inputs=img, outputs=outputs)
    print("\n\n\n\n", Encoder.summary())

    return Encoder



def Decoder(input_shape):
    inputs = Input(shape=input_shape)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    outputs = Conv2D(1, (1, 1), activation='relu', padding='same')(x)



    Decoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    print("\n\n\n\n", Decoder.summary())

    return Decoder















def CAE(input_shape):
        inputs = Input(shape=input_shape)
        inputs_noise = GaussianNoise(stddev=0.01)(inputs)

        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)(inputs_noise)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        pass_feature = Conv2D(128, (3, 3), activation='relu', padding='same')(x)




        x = MaxPooling2D(pool_size=(2, 2))(pass_feature)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (1, 1), activation='relu', padding='same')(x)

        # passthrough layer - end
        x = PassThrough(pass_feature=pass_feature)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)


        # Decoder
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64,  (1, 1), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        outputs = Conv2D(1, (1, 1), activation='relu', padding='same')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(model.summary())

        return model




######################################################################################################




















































