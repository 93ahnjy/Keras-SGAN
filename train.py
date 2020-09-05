import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Conv2D,BatchNormalization, Dense, MaxPooling2D, Flatten, Dropout, Input, Activation
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from CNN_model import vgg16, Resnet50


import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os







def one_hot_encoding(class_list, class_n):
    one_hot = np.array(class_list) == class_n
    one_hot = one_hot.astype(np.uint8)

    return one_hot




def generate_grad_cam(img_tensor, model, class_index, activation_layer):
    """
    params:
    -------
    img_tensor: resnet50 모델의 이미지 전처리를 통한 image tensor
    model: pretrained resnet50 모델 (include_top=True)
    class_index: 이미지넷 정답 레이블
    activation_layer: 시각화하려는 레이어 이름

    return:
    grad_cam: grad_cam 히트맵
    """
    inp = model.input
    y_c = model.output.op.inputs[0][0, class_index]
    A_k = model.get_layer(activation_layer).output


    ## 이미지 텐서를 입력해서
    ## 해당 액티베이션 레이어의 아웃풋(a_k)과
    ## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
    get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [conv_output, grad_val, model_output] = get_output([img_tensor])



    ## 배치 사이즈가 1이므로 배치 차원을 없앤다.
    conv_output = conv_output[0]
    grad_val = grad_val[0]



    ## 구한 gradient를 픽셀 가로세로로 평균내서 a^c_k를 구한다.
    weights = np.mean(grad_val, axis=(0, 1))



    ## 추출한 conv_output에 weight를 곱하고 합하여 grad_cam을 얻는다.
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]

    grad_cam = cv2.resize(grad_cam, (32, 32))

    ## ReLU를 씌워 음수를 0으로 만든다.
    grad_cam = np.maximum(grad_cam, 0)
    print("grad_cam max", grad_cam.max())
    grad_cam = grad_cam / (grad_cam.max() + 0.0001)

    return grad_cam










class MyModel(tf.keras.Model):

    def __init__(self, num_classes, input_size, backbone, pretrained=True, saved_model_path=None):
        self.num_classes = num_classes
        self.input_size  = input_size


        if self.num_classes > 1:
            self.class_mode = 'categorical'
            self.loss       = 'categorical_crossentropy'
            self.last_activation = 'softmax'

        else:
            self.class_mode = 'binary'
            self.loss       = 'binary_crossentropy'
            self.last_activation = 'sigmoid'

        super(MyModel, self).__init__(name='my_model')



        if backbone == "vgg16":
            self.model = vgg16(input_size = self.input_size,
                               num_classes= self.num_classes,
                               pretrained = pretrained)
        elif backbone == 'resnet50':
            self.model = Resnet50(input_size = self.input_size,
                                  num_classes= self.num_classes)


        if saved_model_path:
            self.model.load_weights(saved_model_path)
            print("Loaded model", saved_model_path, "from disk")






    def prepare_dataset(self, train_dir, test_dir,  train_batch_size, test_batch_size):

        train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1, zca_epsilon=1e-06)
        test_datagen  = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(train_dir,
                                                         target_size=self.input_size,
                                                         batch_size=train_batch_size,
                                                         class_mode=self.class_mode)
        x, y = training_set.next()
        print(x.shape, y.__len__())

        test_set = test_datagen.flow_from_directory(test_dir,
                                                    target_size=self.input_size,
                                                    batch_size=test_batch_size,
                                                    class_mode=self.class_mode)
        x, y = test_set.next()
        print(x.shape, y.__len__())

        return training_set, test_set






    def train(self, train_dir, test_dir, epoch,  train_batch_size, test_batch_size):

        training_set, test_set = self.prepare_dataset(train_dir,
                                                      test_dir,
                                                      train_batch_size,
                                                      test_batch_size)


        optimizer = optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0011)


        self.model.compile(loss=self.loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])
        print("loss type :", self.loss)


        self.model.fit_generator(training_set,
                                 epochs=epoch,
                                 validation_data=test_set)


        self.model.save_weights("saved_model.h5")
        self.test_model(test_dir)




    def test_model(self, test_dir):

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_set = test_datagen.flow_from_directory(test_dir,
                                                    target_size=self.input_size,
                                                    batch_size=1,
                                                    class_mode=self.class_mode)

        x, y = test_set.next()
        predictions = self.model.predict(x)
        class_index = np.argmax(predictions)

        print(predictions, predictions.shape, class_index)



        grad_cam = generate_grad_cam(x, self.model, class_index, "")
        print(grad_cam.shape)


        plt.imshow(x[0])
        plt.imshow(grad_cam, cmap='jet', alpha=0.5)
        plt.show()








print("\n\n\n\nUsing custom model")




model = MyModel(num_classes=10,
                input_size=(32, 32),
                backbone="resnet50",
                saved_model_path=None
                )

model.train('./cifar10/train', 
            './cifar10/test',
            train_batch_size=128,
            test_batch_size=10000,
            epoch=1)
# optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.test_model('./cifar10/test')
'''




model = MyModel(num_classes=1,
                input_size=(128, 128),
                backbone="resnet50",
                saved_model_path =None)

model.train('./Surface_defect_dataset/Train_crop',
            './Surface_defect_dataset/Test_crop',
            train_batch_size=8,
            test_batch_size=420,
            epoch=100)
# optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


model.test_model('./Surface_defect_dataset/Test_crop')
'''





