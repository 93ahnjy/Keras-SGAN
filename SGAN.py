from __future__ import print_function, division

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Activation, BatchNormalization, Activation, Add, ZeroPadding2D, LeakyReLU, Layer, UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, LeakyReLU, Input, Dense, Reshape, Flatten, Dropout, concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


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


    if test_dir:
        test_set = test_datagen.flow_from_directory(test_dir,
                                                    target_size=input_size,
                                                    batch_size=test_batch_size,
                                                    color_mode=color_mode,
                                                    class_mode=class_mode)
        return training_set, test_set

    else:
        return training_set












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
        model.add(Conv2D(output_channel, kernel_size=kernel_size, padding="same", strides=2))
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
    x = BatchNormalization(momentum=0.8)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    pass_feature = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(pass_feature)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)

    # passthrough layer - end
    # x = PassThrough(pass_feature=pass_feature)(x)
    outputs = MaxPooling2D(pool_size=(2, 2))(x)


    Encoder = Model(inputs=img, outputs=outputs)

    return Encoder



def Decoder(input_shape):
    inputs = Input(shape=input_shape)

    # Decoder
    '''
    x = Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape)(inputs)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D(size=(2, 2))(x)
    '''
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
    #print("\n\n\n\n", Decoder.summary())

    return Decoder














class CAE():

    def __init__(self, setting):

        self.setting = setting

        # Setting training figure
        self.train_dir        = self.setting['CAE_setting']['train_dir']
        self.test_dir         = self.setting['CAE_setting']['test_dir']
        self.train_batch_size = self.setting['CAE_setting']['train_batch_size']
        self.test_batch_size  = self.setting['CAE_setting']['test_batch_size']
        self.epochs           = self.setting['CAE_setting']['epochs']
        self.save_model_name  = self.setting['CAE_setting']['save_model_name']

        self.img_shape = setting['image_shape']
        self.img_rows  = self.img_shape[0]
        self.img_cols  = self.img_shape[1]
        self.channels  = self.img_shape[2]


        # Build model
        self.encoder = Encoder(input_shape=self.img_shape)
        self.decoder = Decoder(self.encoder.output.shape[1:])

        self.Conv_AutoEncoder = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output))



        # Setting loss / optimizer & summary
        self.Conv_AutoEncoder.compile(loss='mse',
                                      optimizer=Adam(lr=1e-3))

        '''
        model = Sequential([self.encoder, self.decoder])

        img    = Input(shape=self.img_shape)
        re_img = model(img)
        self.Conv_AutoEncoder = Model(img, re_img)
        '''








    def train(self):

        class_mode = 'grayscale' if self.channels == 1  else 'rgb'


        train_set, test_set = prepare_dataset(input_size=(self.img_shape[0], self.img_shape[1]),
                                              train_dir=self.train_dir,
                                              test_dir=self.test_dir,
                                              train_batch_size=self.train_batch_size,
                                              test_batch_size=self.test_batch_size,
                                              class_mode='input',
                                              color_mode=class_mode
                                              )

        self.Conv_AutoEncoder.fit_generator(train_set,
                                            epochs=self.epochs,
                                            validation_data=test_set)


        self.Conv_AutoEncoder.save(self.save_model_name)





        '''
        for step, real_img in enumerate(train_set):
            loss     = self.Conv_AutoEncoder.train_on_batch(real_img, real_img)
        '''

        fig = plt.figure()
        for i in range(10):

            img, _ = train_set.next()
            re_img = self.Conv_AutoEncoder.predict(np.expand_dims(img[0], 0))

            print(img[0].shape)

            if self.channels == 1:
                fig.add_subplot(1, 2, 1)
                plt.imshow(img[0].squeeze(axis=2), cmap=pylab.gray())
                fig.add_subplot(1, 2, 2)
                plt.imshow(re_img.squeeze(axis=(0, 3)), cmap=pylab.gray())
            else:
                fig.add_subplot(1, 2, 1)
                plt.imshow(img[0])
                fig.add_subplot(1, 2, 2)
                plt.imshow(re_img.squeeze(axis=0))



            plt.savefig("CAE_result_{}png.".format(str(i)))

            plt.show(block=False)
            plt.pause(1)





        print("\n\n\n\n\n\n\n\n\n\n\n\n\n")









########################################################################################################################





class GAN():
    def __init__(self, setting):


        self.setting = setting

        # Input shape
        self.img_shape = setting['image_shape']
        self.img_rows = self.img_shape[0]
        self.img_cols = self.img_shape[1]
        self.channels = self.img_shape[2]


        self.latent_dim = setting['generator_setting']['input_dim']
        self.gen_dim    = setting['generator_setting']['init_filter_size']
        self.upsample   = setting['generator_setting']['upsampling']                # deconv,      nearest
        self.donwsample = setting['discriminator_setting']['downsampling']          # conv_stride, maxpool

        self.load_model_name = setting['discriminator_setting']['load_model_name']

        self.train_dir  = setting['SGAN_setting']['train_dir']
        self.test_dir  = setting['SGAN_setting']['test_dir']
        self.batch_size = setting['SGAN_setting']['batch_size']

        self.num_class  = setting['SGAN_setting']['num_class']


        optimizer = Adam(lr=0.0002, beta_1=0.5, decay=0.00005)


        # ---------------------
        #  Setting Discriminator
        # ---------------------


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                                   loss_weights=[0.5, 0.5],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])





        # ---------------------
        #  Setting Generator
        # ---------------------

        # Build the generator
        self.generator = self.build_generator()




        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)




        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        print(self.discriminator.summary())



        # The discriminator takes generated images as input and determines validity
        # Discriminator has two outputs - 'valid', 'classes'. GAN uses only 'valid' output
        valid, _ = self.discriminator(img)




        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)




    def build_generator(self):


        model = Sequential()

        # Generator
        '''
        model.add(Dense(self.gen_dim * self.img_rows//16 * self.img_cols//16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows//16, self.img_cols//16, self.gen_dim)))

        model.add(Conv2D(256, (3, 3), padding='same'))
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

        model.add(Conv2D(self.channels, (1, 1), activation='tanh', padding='same'))
        '''


        '''
        # 2nd model
        model.add(Dense(self.gen_dim * self.img_rows//4 * self.img_cols//4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows//4, self.img_cols//4, self.gen_dim)))

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        add_Conv_Upsampling(model, 64, 3, self.upsample)
        add_Conv_Upsampling(model, 32, 3, self.upsample)

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        '''




        # 3rd model
        model.add(Dense(self.gen_dim * self.img_rows//4 * self.img_cols//4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows//4, self.img_cols//4, self.gen_dim)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)





    def build_discriminator(self):

        '''
        Check and Try other architectures
        '''



        '''
        # Samsung project - Load encoder to make discriminator
        
        encoder = Encoder(self.img_shape)

        CAE_saved = load_model(self.load_model_name)
        CAE_saved.summary()

        for i in range(len(encoder.layers)):
            encoder.layers[i].set_weights(CAE_saved.layers[i].get_weights())

        model = Sequential()
        model.add(encoder)
        model.add(Flatten())
        '''

        # keras sgan
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.summary()



        img = Input(shape=self.img_shape)
        features = model(img)

        # classify unlabeled <-> generated
        valid = Dense(1, activation='sigmoid')(features)

        # classify label of labeled img
        label = Dense(self.num_class + 1, activation='softmax')(features)


        return Model(img, [valid, label])














    def train(self):

        # Load decoder weight to GAN's discriminator
        # self.combined.summary()
        # self.combined = load_model('CAE_weights.h5')


        color_mode = 'grayscale' if self.channels == 1 else 'rgb'

        train_datagen = ImageDataGenerator(rescale=1./255)

        train_set     = train_datagen.flow_from_directory(self.train_dir,
                                                          target_size=[self.img_rows, self.img_cols],
                                                          batch_size=self.batch_size,
                                                          color_mode=color_mode,    # 'rgb'
                                                          class_mode='binary')

        test_set     = train_datagen.flow_from_directory(self.test_dir,
                                                          target_size=[self.img_rows, self.img_cols],
                                                          batch_size=self.batch_size,
                                                          color_mode=color_mode,    # 'rgb'
                                                          class_mode='binary')

        epoch = 0
        i = 0

        for x_batch, y_label in train_set:



            x_batch = x_batch - 0.5
            y_label = y_label.reshape(-1, 1)


            # For N class, real / fake has 1:1
            half_batch = self.batch_size // 2
            cw1 = {0: 1, 1: 1}

            # For N class, each class has 'K/N' images while there are total 'K' fake images
            # So, each 'real img' has 'N' times more weight than 'fake img' (--> weight = 1 / frequency)
            cw2 = {i:  self.num_class / half_batch for i in range(self.num_class)}
            cw2[self.num_class] = 1 / half_batch



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


            # One-hot encoding of labels
            # fake_label will be one-hot encoded to 'N+1'th column
            labels      = to_categorical(y_label, num_classes=self.num_class+1)
            fake_labels = to_categorical(np.full((x_batch.shape[0], 1), self.num_class), num_classes=self.num_class+1)


            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x_batch, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)









            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            self.discriminator.trainable = False


            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[cw1, cw2])


            # Plot the progress
            d_lr = K.eval(self.discriminator.optimizer.lr)
            g_lr = K.eval(self.combined.optimizer.lr)


            print ("epoch : %d   step : %d / %d    [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]  [d_lr : %f]  [g_lr :%f]"
                 % (epoch, i, train_set.__len__(), d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss, d_lr, g_lr))


            # If at save interval => save generated image samples
            if (i + 1) % 20 == 0:
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



            if (i + 1) % 100 == 0:
                plt.savefig("result_{}_{}.png.".format(str(epoch), str(i)))
                plt.close()

                # Test the discriminator
                x_batch_t, y_labels_t = test_set.next()
                labels_t = to_categorical(y_labels_t, num_classes=self.num_class + 1)

                print(self.discriminator.metrics_names)
                print(self.discriminator.evaluate(x_batch_t, [valid, labels_t], verbose=1))


            if (i + 1) % train_set.__len__() == 0:
                epoch += 1
                i = 0

            i += 1





if __name__ == '__main__':


    setting = {'image_shape' : (64, 64, 1),

               'CAE_setting' : {'train_dir'         : './Surface_defect_dataset/mnist_png/train',  #
                                'test_dir'          : './Surface_defect_dataset/mnist_png/test',
                                'train_batch_size'  : 8,
                                'test_batch_size'   : 420,
                                'epochs'            : 1,
                                'save_model_name'   : 'CAE_weights.h5'},

               'generator_setting'     : {'input_dim'        : 150,
                                          'init_filter_size' : 256,
                                          'upsampling'       : 'nearest'},

               'discriminator_setting' : {'downsampling'     : 'maxpool',
                                          'load_model_name'  : 'CAE_weights.h5'},

               'SGAN_setting'          : {'train_dir'  : './Surface_defect_dataset/mnist_png/train',
                                          'test_dir': './Surface_defect_dataset/mnist_png/test',
                                          'num_class'  : 10,
                                          'batch_size' : 128,
                                          'epochs'     : 30000}
               }


    cae = CAE(setting)
    cae.train()


    gan = GAN(setting)
    gan.train()   # Train_crop_gray









