""" This function is used to train a neural network model to find phone. """

# Loading dependencies.
import sys
import os
import time
from tensorflow import keras
# Data pre-processing libraries.
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Machine learning modelling libraries.
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dropout, Dense


def train_phone_finder(path):
    """ This funtion is used to train the model. """
    start = time.time()
    os.chdir(path)
    cwd = os.getcwd()
    label_data = []

    # Accessing data files.
    for file in os.listdir(cwd):
        if file.endswith(".txt"):
            with open("labels.txt") as file:
                for line in file:
                    data_line = [l.strip() for l in line.split(' ')]
                    label_data.append(data_line)

    # Processing Image files.
    x_variable = []
    y_variable = []

    for label in label_data:
        img = cv2.imread(label[0])
        # Resized image
        resized_image = cv2.resize(img, (128, 128))
        x_variable.append(resized_image.tolist())
        y_variable.append([float(label[1]), float(label[2])])

        # Bright and contrast Image
        bright_image = cv2.convertScaleAbs(resized_image, alpha=1.5 , beta=40 )
        x_variable.append(bright_image.tolist())
        y_variable.append([float(label[1]), float(label[2])])

        # gaussian smoothing
        smoothed_image = cv2.GaussianBlur(resized_image,(3,3),0)
        x_variable.append(smoothed_image.tolist())
        y_variable.append([float(label[1]), float(label[2])])

        # Median filtering
        median_img = cv2.medianBlur(resized_image, 7)
        x_variable.append(median_img.tolist())
        y_variable.append([float(label[1]), float(label[2])])

        # Flipping of images
        flip_image_x = cv2.flip(resized_image, 0)
        flip_image_y = cv2.flip(resized_image, 1)

        if float(label[1]) > 0.5:
            x_variable.append(flip_image_y.tolist())
            y_variable.append([1 - float(label[1]), float(label[2])])
        else:
            x_variable.append(flip_image_y.tolist())
            y_variable.append([float(label[1]), float(label[2])])

        if float(label[2]) > 0.5:
            x_variable.append(flip_image_x.tolist())
            y_variable.append([float(label[1]), 1 - float(label[2])])
        else:
            x_variable.append(flip_image_x.tolist())
            y_variable.append([float(label[1]), float(label[2])])

    x_variable = np.asarray(x_variable)
    y_variable = np.asarray(y_variable)

    # Rescaling pixel values between 0 and 1.
    x_variable = np.interp(x_variable, (x_variable.min(), x_variable.max()), (0, 1))

    # Splitting data into training and test sets.
    (train_x, test_x, train_y, test_y) = train_test_split(x_variable, y_variable,
                                                          test_size=0.22)

    # Model input parameters.
    height = x_variable.shape[-2]
    width = x_variable.shape[-3]
    depth = x_variable.shape[-1]
    input_shp = (height, width, depth)

    # Deep Learning Architecture.
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=input_shp))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("sigmoid"))

    model.add(Dense(128))
    model.add(Activation("sigmoid"))

    model.add(Dense(y_variable.shape[-1]))
    epochs = 60

    model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])

    print("\n\n\nTraining begins ...\n\n\n")
    history_object = model.fit(train_x, train_y, epochs=epochs, verbose=2, batch_size=16,
                               validation_data=(test_x, test_y))

    model.save('train_phone_finder_weights.h5')

    print('Generating loss chart...')
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('model.png')


    # Done
    print('Done.')
    print("\n\n\nTraining complete. \n\n\n")
    print("Program Runtime {0} seconds, Exiting ...".format(time.time() - start))


def main():
    """ This function is used to run the program. """
    train_phone_finder(sys.argv[1])


if __name__ == "__main__":
    main()
