# importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# export the data
# directory
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken",
             "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane",
             "cavallo": "horse", "elephant": "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto",
             "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}

# X is the array of images
# y is the labels
X = []  # images
y = []  # labels
i = 0  # variable to name labels.. since we have different names such as "cane", "dog" we gotta assign numbers to differentiate

# iterate over files in that directory
for filename in os.listdir('raw-img'):
    e = 0  # for testing purposes since we only want some picture to see output
    animalFolder = os.path.join('raw-img', filename)
    print("MainFolder : {}".format(animalFolder))
    # checking if it is a file
    for picture in os.listdir(animalFolder):
        e += 1
        pathPicture = os.path.join(animalFolder, picture)
        print(pathPicture)
        img = mpimg.imread(pathPicture)
        # resized the image to 128x 128 since the images size varies on the folder provided
        resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        # resized_img = tf.cast(resized_img, np.uint8)

        # some images are corrupted. Therefore a validation set is necessary to avoid error
        # populating the matrix of each image into X and the label into y accordingly
        if resized_img.shape == (128, 128, 3):
            X.append(resized_img)
            y.append(i)
        # for testing purposes only since we have huge dataset of images
        if e == 10:
            break
    i += 1

# #convert list to numpy array
X = np.array(X)
y = np.array(y)
print("hey there")
print(type(X), X.shape)
print(type(y), y.shape)
