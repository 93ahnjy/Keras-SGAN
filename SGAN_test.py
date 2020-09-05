from __future__ import print_function, division

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Activation, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, Layer, UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, LeakyReLU, Input, Dense, Reshape, Flatten, Dropout, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import TruncatedNormal


import pylab
import matplotlib.pyplot as plt
import cv2
import numpy as np









def prepare_dataset(input_size, train_dir, test_dir, train_batch_size, test_batch_size, class_mode, color_mode):
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, width_shift_range=0.1,
                                       height_shift_range=0.1, zca_epsilon=1e-06)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_dir,
                                                     target_size=input_size,
                                                     batch_size=train_batch_size,
                                                     color_mode=color_mode,
                                                     class_mode=class_mode)


    test_set = test_datagen.flow_from_directory(test_dir,
                                                target_size=input_size,
                                                batch_size=test_batch_size,
                                                color_mode=color_mode,
                                                class_mode=class_mode)
    return training_set, test_set












def add_Conv_Upsampling(model, output_channel, kernel_size, mode):

    if mode == 'deconv':
        model.add(Conv2DTranspose(output_channel, kernel_size=kernel_size, padding="same", strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    elif mode == 'nearest':
        model.add(UpSampling2D())
        model.add(Conv2D(output_channel, kernel_size=kernel_size, padding="same", strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))




def add_Conv_Downsampling(model, output_channel, kernel_size, mode):

    if mode == 'conv_stride':
        model.add(Conv2D(output_channel, kernel_size=kernel_size, padding="valid", strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

    elif mode == 'maxpool':
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(output_channel, kernel_size=kernel_size, padding="same", strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))






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




def Encoder(input_shape):
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


    Encoder = Model(inputs=img, outputs=outputs)
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



    Decoder = Model(inputs=inputs, outputs=outputs)
    print("\n\n\n\n", Decoder.summary())

    return Decoder



class CAE():

    def __init__(self, img_shape):

        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.channels = img_shape[2]
        self.img_shape = img_shape


        self.encoder = Encoder(input_shape=self.img_shape)
        self.decoder = Decoder(self.encoder.output.shape[1:])



        self.Conv_AutoEncoder = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output))



        self.Conv_AutoEncoder.compile(loss='mse',
                                      optimizer=Adam(lr=1e-3))
        self.Conv_AutoEncoder.summary()




    def train(self, setting):
        train_set, test_set = prepare_dataset(input_size=(self.img_shape[0], self.img_shape[1]),
                                              train_dir=setting['train_dir'],
                                              test_dir=setting['test_dir'],
                                              train_batch_size=setting['train_batch_size'],
                                              test_batch_size=setting['test_batch_size'],
                                              class_mode=setting['class_mode'],
                                              color_mode=setting['color_mode']
                                              )

        self.Conv_AutoEncoder.fit_generator(train_set,
                                            epochs=20,
                                            validation_data=test_set)
        '''
        for step, real_img in enumerate(train_set):
            loss     = self.Conv_AutoEncoder.train_on_batch(real_img, real_img)
        '''

        for i in range(10):

            img, _ = train_set.next()
            img = self.Conv_AutoEncoder.predict(np.expand_dims(img[0], 0))

            plt.imshow(img.squeeze(axis=(0,3)), cmap=pylab.gray())
            plt.show()




        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")






setting = {'train_dir' : './Surface_defect_dataset/faces',
           'test_dir'  : './Surface_defect_dataset/faces',
           'train_batch_size' : 8,
           'test_batch_size'  : 420,
           'epochs'           : 5,
           'class_mode' : 'input',
           'color_mode' : 'grayscale'

}
#cae = CAE(img_shape=(128, 128, 1))
#cae.train(setting)





########################################################################################################################





class GAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape  = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.gen_dim  = 256
        self.upsample   = 'nearest'   # deconv,      nearest
        self.donwsample = 'maxpool'   # conv_stride, maxpool


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

        model.add(Dense(self.gen_dim * self.img_rows//4 * self.img_cols//4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows//4, self.img_cols//4, self.gen_dim)))

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        add_Conv_Upsampling(model, 64, 3, self.upsample)
        add_Conv_Upsampling(model, 32, 3, self.upsample)

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))



        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)



    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        add_Conv_Downsampling(model, 64,  3, self.donwsample)
        add_Conv_Downsampling(model, 128, 3, self.donwsample)
        add_Conv_Downsampling(model, 256, 3, self.donwsample)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))



        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)






    def train(self, epochs, dataset_dir, batch_size=128):

        if self.channels == 3:
            color_mode = 'rgb'
        else:
            color_mode = 'grayscale'


        train_datagen = ImageDataGenerator(rescale=1./255)

        train_set     = train_datagen.flow_from_directory(dataset_dir,
                                                          target_size=[self.img_rows, self.img_cols],
                                                          batch_size=batch_size,
                                                          color_mode=color_mode,    # 'rgb'
                                                          class_mode=None)



        epoch = 0
        i = 0

        for step, x_batch in enumerate(train_set):



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
            print ("epoch : %d   step : %d / %d    [D loss: %f, acc.: %.2f%%] [G loss: %f]  [d_lr : %f]  [g_lr :%f]"
                 % (epoch, i, train_set.__len__(), d_loss[0], 100*d_loss[1], g_loss, d_lr, g_lr))




            # If at save interval => save generated image samples
            if (step + 1) % 20 == 0:
                noise = np.random.uniform(-1, 1, (x_batch.shape[0], self.latent_dim))

                img = self.generator.predict(noise)
                img = 0.5 * img + 0.5

                fig = plt.gcf()
                fig.set_size_inches(3, 3)


                if self.channels == 1:
                    img = img[0].squeeze(axis=2)
                    #img = cv2.GaussianBlur(img, (3, 3), 0)
                    plt.imshow(img, cmap=pylab.gray())  # if gray scale :img[0].squeeze(axis=2), cmap=pylab.gray()
                else:
                    plt.imshow(img[0])

                plt.show(block=False)
                plt.pause(0.001)



            if (step + 1) % 100 == 0:
                plt.savefig("result_test_{}.png.".format(str(step)))
                plt.close()


            if (i + 1) == train_set.__len__():
                epoch += 1
                i = 0

            i += 1








if __name__ == '__main__':

    i=0



    gan = GAN()
    #cae = CAE(img_shape=(32,32,3))

    #print(cae.encoder)
    #print(gan.generator)
    gan.train(epochs=30000, batch_size=32, dataset_dir='./Surface_defect_dataset/faces')   # Train_crop_gray





















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


'''
Conv2DTranspose


'''