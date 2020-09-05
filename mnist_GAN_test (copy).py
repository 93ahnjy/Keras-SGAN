from __future__ import print_function, division

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Activation, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, LeakyReLU, Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import TruncatedNormal

from CNN_model import CAE, PassThrough, Encoder, Decoder

import pylab
import matplotlib.pyplot as plt


import sys

import numpy as np

class GAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5, decay=0.00005)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same", kernel_initializer='random_normal'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same", kernel_initializer='random_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer='random_normal'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)






    def train(self, epochs, batch_size=128):

        if self.channels == 3:
            color_mode = 'rgb'
        else:
            color_mode = 'grayscale'


        train_datagen = ImageDataGenerator(rescale=1./255)

        train_set     = train_datagen.flow_from_directory('./Surface_defect_dataset/faces',
                                                          target_size=[self.img_rows, self.img_cols],
                                                          batch_size=batch_size,
                                                          color_mode=color_mode,    # 'rgb'
                                                          class_mode=None)


        epoch = 0
        i = 0

        for epoch in range(epochs):

            for step, x_batch in enumerate(train_set):


                '''
                print(x_batch.shape, i)
                if i > 20000:
                    plt.imshow(x_batch[0].squeeze(axis=2), cmap=pylab.gray())
                    plt.show(block=False)
                    plt.pause(0.01)
    
                    if i % 10 == 0:
                        plt.close()
    
                '''
                x_batch = x_batch - 0.5

                # Adversarial ground truths
                # Use x_batch.shape[0] instead of batch_size because of last batch's size ( < batch_size)


                valid = np.ones((x_batch.shape[0], 1))
                fake = np.zeros((x_batch.shape[0], 1))

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.discriminator.trainable = True
                noise = np.random.uniform(-1, 1, (x_batch.shape[0], self.latent_dim))


                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)


                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(x_batch, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)





                # ---------------------
                #  Train Generator
                # ---------------------


                # Train the generator (to have the discriminator label samples as valid)
                self.discriminator.trainable = False
                g_loss = self.combined.train_on_batch(noise, valid)



                # Plot the progress
                d_lr = K.eval(self.discriminator.optimizer.lr)
                g_lr = K.eval(self.discriminator.optimizer.lr)
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]  [d_lr : %f]  [g_lr :%f]" % (step, d_loss[0], 100*d_loss[1], g_loss, d_lr, g_lr))




                # If at save interval => save generated image samples
                if (step + 1) % 20 == 0:
                    noise = np.random.uniform(-1, 1, (x_batch.shape[0], self.latent_dim))

                    img = self.generator.predict(noise)
                    img = 0.5 * img + 0.5

                    fig = plt.gcf()
                    fig.set_size_inches(3, 3)




                    plt.imshow(img[0].squeeze(axis=2), cmap=pylab.gray())  # if gray scale :img[0].squeeze(axis=2), cmap=pylab.gray()
                    plt.show(block=False)
                    plt.pause(0.01)


                if (step + 1) % 100 == 0:
                    plt.savefig("result_{}.png.".format(str(step)))
                    plt.close()


                if (step + 1) == epochs:
                    break








if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32)





















'''
     def build_generator(self):


        model = Sequential()

        # Decoder
        model.add(Conv2D(128, (3, 3), padding='same', input_shape=self.n_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (1, 1), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (1, 1), activation='tanh', padding='same'))


        print("\n\n\nGenerator")
        model.summary()

        noise = Input(shape=self.n_shape)
        img = model(noise)


        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # Encoder
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))


        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))



        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))



        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))



        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        # pass_feature = Conv2D(128, (3, 3), activation='relu', padding='same')(x)


        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))


        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (1, 1), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # passthrough layer - end
        # x = PassThrough(pass_feature=pass_feature)(x)


        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))


        print("\n\n\nDiscriminator")
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

'''








'''
    def build_generator(self):


        model = Sequential()

        # Decoder
        model.add(Conv2D(128, (3, 3), padding='same', input_shape=self.n_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (1, 1), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (1, 1), activation='tanh', padding='same'))


        print("\n\n\nGenerator")
        model.summary()

        noise = Input(shape=self.n_shape)
        img = model(noise)


        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # Encoder
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))


        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))



        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        # pass_feature = Conv2D(128, (3, 3), activation='relu', padding='same')(x)


        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))



        model.add(MaxPooling2D(pool_size=(2, 2)))



        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (1, 1), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # passthrough layer - end
        # x = PassThrough(pass_feature=pass_feature)(x)


        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))


        print("\n\n\nDiscriminator")
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


'''







'''
    def build_generator(self):


        model = Sequential()

        # Decoder
        model.add(Conv2DTranspose(512, (4, 4), strides=1, padding='same', input_shape=self.n_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))


        model.add(Conv2DTranspose(256, (4, 4),  strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))


        model.add(Conv2DTranspose(128, (4, 4),  strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))


        model.add(Conv2DTranspose(64, (4, 4),  strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(self.channels, (1, 1), strides=2, activation='tanh', padding='same'))


        print("\n\n\nGenerator")
        model.summary()

        noise = Input(shape=self.n_shape)
        img = model(noise)

        return Model(noise, img)







    def build_discriminator(self):

        model = Sequential()

        # Encoder
        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=self.img_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))


        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))



        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))


        # passthrough layer - end
        # x = PassThrough(pass_feature=pass_feature)(x)


        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(1, kernel_size=4, strides=1, activation='sigmoid', padding="valid"))
        model.add(Reshape((1, )))


        print("\n\n\nDiscriminator")
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
'''














'''

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same", kernel_initializer='random_normal'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same", kernel_initializer='random_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer='random_normal'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer='random_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)





'''