import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D,BatchNormalization, Dense, MaxPooling2D, Flatten, Dropout, Input, Activation, GaussianNoise
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from CNN_model import CAE, PassThrough, Encoder, Decoder
import os
import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt


def prepare_dataset(input_size, class_mode, train_dir, test_dir, train_batch_size, test_batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, width_shift_range=0.1,
                                       height_shift_range=0.1, zca_epsilon=1e-06)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_dir,
                                                     target_size=input_size,
                                                     batch_size=train_batch_size,
                                                     color_mode='grayscale',
                                                     class_mode=class_mode)


    test_set = test_datagen.flow_from_directory(test_dir,
                                                target_size=input_size,
                                                batch_size=test_batch_size,
                                                color_mode='grayscale',
                                                class_mode=class_mode)
    return training_set, test_set











class Conv_AutoEncoder(tf.keras.Model):

    def __init__(self, input_shape):
        super(Conv_AutoEncoder, self).__init__()
        self.model        = CAE(input_shape=input_shape)
        self.input_shapes = input_shape
        self.input_size   = (input_shape[0], input_shape[1])







    def train(self, train_dir, test_dir, epochs, train_batch_size, test_batch_size):

        training_set, test_set = prepare_dataset(self.input_size,
                                                 'input',
                                                 train_dir,
                                                 test_dir,
                                                 train_batch_size,
                                                 test_batch_size)

        optimizer = tf.keras.optimizers.Adam(lr=1e-3)

        self.model.compile(loss='mse',
                           optimizer=optimizer)

        self.model.fit_generator(training_set,
                                 epochs=epochs,
                                 validation_data=test_set)

        self.model.save_weights("saved_model_AE.h5")



    def test_model(self, test_dir):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_set = test_datagen.flow_from_directory(test_dir,
                                                    target_size=self.input_size,
                                                    batch_size=210,
                                                    color_mode='grayscale',
                                                    class_mode='input')

        for x_batch_test in test_set:



            for i, x_batch in enumerate(x_batch_test):


                predictions = self.model.predict(np.expand_dims(x_batch[i], axis=0))

                fig, (ax0, ax1) = plt.subplots(1, 2)
                ax0.imshow(cv2.resize(x_batch[i], (128, 128)), cmap=pylab.gray())
                ax1.imshow(cv2.resize(predictions.squeeze(axis=0), (128, 128)), cmap=pylab.gray())

                plt.show()






def image_rescale(img):
    img = 2 * (img - img.min())/(img.max() - img.min()) - 1
    return  img






class SGAN(tf.keras.Model):

    def __init__(self, input_shape):
        super(SGAN, self).__init__()

        self.input_size    = [input_shape[0], input_shape[1]]
        self.cross_entropy = tf.keras.losses.binary_crossentropy




        # Build the generator
        self.generator     = Decoder(input_shape=input_shape)


        # Build and compile the discriminator
        self.discriminator = Encoder(input_shape=self.generator.output.shape[1:])
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=tf.keras.optimizers.Adam(1e-4),
                                   metrics=['accuracy'])



        # The generator takes noise as input and generates imgs
        noise           = Input(shape=(8, 8, 1), name='noise_input', dtype=tf.float32)
        generated_image = self.generator(noise)






        # For the combined model we will only train the generator
        self.discriminator.trainable = False


        # The valid takes generated images as input and determines validity
        decision = self.discriminator(generated_image)


        # Connect the "output layer of the first network(encoder.output)"
        # to the "input layer of the second network".
        self.GAN = tf.keras.Model(inputs=noise, outputs=decision)


        # Trains the generator to fool the discriminator
        self.GAN.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(1e-4))


        print('\n\n\n\n\n')
        print(self.GAN.summary())
        plot_model(self.generator, to_file='generator.png')
        plot_model(self.discriminator, to_file='discriminator.png')



    def train(self, train_dir, test_dir, epochs, train_batch_size, test_batch_size):


        train_set, test_set = prepare_dataset([128, 128],
                                              None,
                                              train_dir,
                                              test_dir,
                                              train_batch_size,
                                              test_batch_size)


        real  = K.ones((train_batch_size, 1))
        fake  = K.zeros((train_batch_size, 1))

        epoch = 0
        for x_batch in train_set:


            noise = K.random_normal_variable(shape=(train_batch_size, self.input_size[0], self.input_size[1], 1), mean=0, scale=1)


            gen_imgs = self.generator.predict(noise, steps=1)
            gen_imgs = image_rescale(gen_imgs)

            # print(gen_imgs[0].shape)
            # plt.imshow(np.squeeze(gen_imgs[0], axis=2), cmap=pylab.gray())
            # plt.show()


            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x_batch, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------


            # NEW noise input
            noise = K.random_normal_variable(shape=(train_batch_size, self.input_size[0], self.input_size[1], 1), mean=0, scale=1)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.GAN.train_on_batch(noise, real)





            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


            if epoch > epochs:
                print("\n\n\nFINISH")
                break

            elif (epoch+1) % 2 == 0:
                noise = K.random_normal_variable(shape=(1, self.input_size[0], self.input_size[1], 1),
                                                 mean=0, scale=1)

                img = self.generator.predict(noise, steps=1)
                img = 0.5 * img + 0.5

                '''
                plt.imshow(img[0].squeeze(axis=2), cmap=pylab.gray())
                plt.show(block=False)
                plt.pause(0.05)
                plt.savefig('result.png')
                '''

                #cv2.imshow("Generator result", img[0])
                #cv2.waitKey(1000)
                #cv2.destroyAllWindows()


            epoch +=1



a = SGAN(input_shape=(8,8,1))
a.train(train_dir='./Surface_defect_dataset/Train_crop_gray',
        test_dir ='./Surface_defect_dataset/Test_crop_gray',
        train_batch_size=32,
        test_batch_size=420,
        epochs=20000)





'''
if __name__ == '__main__':

    model = Conv_AutoEncoder(input_shape=(128, 128, 1))
    model.train(train_dir='./Surface_defect_dataset/Train_crop_gray',
                test_dir ='./Surface_defect_dataset/Test_crop_gray',
                train_batch_size=8,
                test_batch_size=420,
                epochs=1)

    model.test_model(test_dir='./Surface_defect_dataset/Test_crop_gray')
'''

