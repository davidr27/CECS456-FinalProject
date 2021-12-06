# importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.model_selection import train_test_split

print(tf.__version__)
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
##from keras.utils import to_catergorical
from keras.utils.np_utils import to_categorical

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
        # print(pathPicture)
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
        # if e == 10:
        #     break
    i += 1

# #convert list to numpy array
X = np.array(X)
y = np.array(y)
print("X numpy array: ", type(X), X.shape)
print("y numpy array: ", type(y), y.shape)
# we already obtained the images in our X array and our label in our y array
# we might need to do the transfer learning .....
# we need to research transfer learning for the next part of the project
# after transfer learning
# Load data set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
print("X_train: ", x_train.shape)
print("X_val: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_val: ", y_test.shape)

# Normalize data set to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create a model and add layers
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Print a summary of the model
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test), shuffle=True)

# Save neural network structure
model_structure = model.to_json()
f = open("model_structure.json", 'w')
f.write(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")
