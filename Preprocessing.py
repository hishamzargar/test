
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

#Path of the data. My folder name is data and I am working in the same directory
directory = "data"

#Different characters of the language
classes = ["Aen", "Alif", "Bay", "ChotiYe", "Daal",
	      "Gaaf", "Hamza", "Jeem"]

# The size of the images
image_size = 28

# All images are checked in the data folder one by one using x variable
for x in classes :
	path = os.path.join(directory, x)
	for image in os.listdir(path):
		arr_of_img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

np_arr_data = []

def data_create():
	for x in classes :
		path = os.path.join(directory, x)
		each_class = classes.index(x)
		for img in os.listdir(path):
			try :                                                                        #exception handling
				arr_of_img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE) #using OpenCV2 library
				np_array = cv2.resize(arr_of_img, (image_size, image_size))              #resizing the image according to specified image size
				np_arr_data.append([np_array, each_class])
			except Exception as e:
				pass

data_create()                       #function call

random.shuffle(np_arr_data)

X = [] #features
y = [] #labels

for features, label in np_arr_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, image_size, image_size, 1)   #reshaping the numpy array

# Dumping out the array data in .pickle file
file_out = open("X.pickle", "wb")
pickle.dump(X, file_out)
file_out.close()

file_out = open("y.pickle", "wb")
pickle.dump(y, file_out)
file_out.close()

file_in = open("X.pickle", "rb")
X = pickle.load(file_in)
