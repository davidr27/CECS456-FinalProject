#importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# export the data
# directory
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken",
             "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane",
             "cavallo": "horse", "elephant": "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto",
             "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}

# setting to current directory

plt.figure(figsize=(20, 20))

# iterate over files in that directory
for filename in os.listdir('raw-img'):
    animalFolder = os.path.join('raw-img', filename)
    print("MainFolder : {}".format(animalFolder))
    # checking if it is a file
    for picture in os.listdir(animalFolder):
        pathPicture = os.path.join(animalFolder, picture)
        print(pathPicture)
        img = mpimg.imread(pathPicture)







