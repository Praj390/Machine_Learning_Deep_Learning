""" This module is used to find coordinates of the phone in the image."""

# Loading dependencies.
import os
import sys

# Data pre-processing libraries.
import cv2
import numpy as np

# Machine learning modelling libraries.
from keras.models import load_model

def find_phone(path):
    """ This function is used to predict coordinates using the trained model. """

    file_name = path
    # Reading the image file.
    x_variable = []
    img = cv2.imread(file_name)
    resized_image = cv2.resize(img, (128, 128))
    x_variable.append(resized_image.tolist())
    x_variable = np.asarray(x_variable)
    x_variable = np.interp(x_variable, (x_variable.min(), x_variable.max()), (0, 1))

    # Loading the Deep Learning Model.
    model = load_model("C:/AuE_Fall_20/Braincorp/find_phone_task_4/find_phone/train_phone_finder_weights.h5")   #Set the path manually for the weights.
    result = model.predict(x_variable)
    print("\n\nPhone in image {0} is located at x-y coordinates given below."
          .format(str(file_name)))
    print("\n{:.4f} {:.4f}".format(result[0][0], result[0][1]))

def main():
    """ This function is used to run the program. """
    find_phone(sys.argv[1])

if __name__ == "__main__":
    main()