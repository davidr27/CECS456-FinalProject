# importing libraries
import os
import cv2
import numpy as np
import joblib
from keras.applications import vgg16
from matplotlib import image as mpimg

images = []
labels = []
i = 0

# iterate over files in that directory
for filename in os.listdir('raw-img'):
    e = 0  # for testing purposes since we only want some picture to see output
    animalFolder = os.path.join('raw-img', filename)
    print("MainFolder : {}".format(animalFolder))
    # checking if it is a file
    for picture in os.listdir(animalFolder):
        e += 1
        pathPicture = os.path.join(animalFolder, picture)
        # print(pathPicture)
        img = mpimg.imread(pathPicture)
        # resized the image to 128x 128 since the images size varies on the folder provided
        resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        # resized_img = tf.cast(resized_img, np.uint8)

        # some images are corrupted. Therefore a validation set is necessary to avoid error
        # populating the matrix of each image into X and the label into y accordingly
        if resized_img.shape == (128, 128, 3):
            images.append(resized_img)
            labels.append(i)
        # for testing purposes only since we have huge dataset of images
        # if e == 10:
        #     break
    i += 1

# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")
