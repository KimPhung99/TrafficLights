from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import load_model
from imutils import paths
import numpy as np
import random
import cv2
import os
import tensorflow as tf


class Model:
    def __init__(self, epoch=60, input_size=(90, 90), clasf=3, datasetname="dataset", modelSaveName="model.h5", learning_rate=0.001, configGPU=True):
        self.des = "Model CNN v1 by PhungHK"
        print(self.des)
        self.datasetname = datasetname
        self.clasf = clasf
        self.input_size = input_size
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.modelSaveName = modelSaveName

        if configGPU:
            self.config()

    def config(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        return tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def pre_data(self, img):
        image = cv2.resize(img, self.input_size)
        image = np.array(image, dtype="float") / 255.0
        return image

    def training(self):
        print("Loading image...")
        imagePaths = list(paths.list_images(self.datasetname))
        random.seed(42)
        random.shuffle(imagePaths)
        # clasf = 3
        labels = []
        data = []
        eye = np.eye(self.clasf, dtype=int)
        print("Eye: {0}".format(eye))

        # loop over the input images
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)

            image = self.pre_data(image)

            data.append(image)
            label = imagePath.split(os.path.sep)[-2]
            labels.append(np.asarray(eye[int(label), :]))

        data = np.array(data)
        print(data.shape)
        labels = np.array(labels)
        print(labels.shape)

        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

        lb = LabelBinarizer()
        trainY = lb.fit_transform(trainY)
        print(trainX.shape)
        testY = lb.transform(testY)

        model = Sequential()

        model.add(Conv2D(32, (7, 7), input_shape=(self.input_size[0], self.input_size[1], 3), padding="SAME", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
        model.add(Conv2D(64, (5, 5), padding="SAME", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="SAME", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.clasf, activation='softmax'))

        print(model.summary())

        opt = SGD(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the network
        print("Training network...")
        history = model.fit(trainX, trainY, epochs=self.epochs, validation_data=(testX, testY))

        # Save the network to disk
        print("Saving model....")
        model.save(self.modelSaveName)
        print("Model saved!")

    def loadding(self):
        self.model = load_model(self.modelSaveName)

    def predict(self, listimg):
        imgs = []
        for im in listimg:
            im = self.pre_data(im)
            imgs.append(im)
        imgs = np.array(imgs)
        return self.model.predict(imgs)
