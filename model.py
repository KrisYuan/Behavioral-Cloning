import csv
import os 
import cv2
import numpy as np
import sklearn
from keras.preprocessing.image import load_img, img_to_array

#read image path from driving_log.csv
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


#save images and steering angles
images = []
angles = []

for line in lines:
    #path for center image, left image and right image
    name_center = './data/IMG/'+line[0].split('/')[-1]
    name_left = './data/IMG/'+line[1].split('/')[-1]
    name_right = './data/IMG/'+line[2].split('/')[-1]
    
    #steering angle for center image, left image and right image
    center_angle = float(line[3])
    correction = 0.2
    left_angle = center_angle + correction
    right_angle =  center_angle - correction
    
    # convert image to an array, save image and angle
    center_image = load_img(name_center)
    center_image = img_to_array(center_image)
    center_image = np.array(center_image)
    images.append(center_image)
    angles.append(center_angle)
        
    left_image = load_img(name_left)
    left_image = img_to_array(left_image)
    left_image = np.array(left_image)
    images.append(left_image)
    angles.append(left_angle)

    right_image = load_img(name_right)
    right_image = img_to_array(right_image)
    right_image = np.array(right_image)
    images.append(right_image)
    angles.append(right_angle)

#get train data
X_train = np.array(images)
y_train = np.array(angles)

#use Keras to build convolutional neural network
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

#initialize model
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(80, 320, 3), output_shape=(80, 320, 3)))

#Convolution1
model.add(Convolution2D(32, 3, 3, input_shape=(80, 320, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

#Convolution2
model.add(Convolution2D(32, 3, 3 ))
model.add(Activation('relu'))
model.add(MaxPooling2D())

#Convolution3
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Flatten())

#FC1
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#FC2
model.add(Dense(16, activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.1))

#Final
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True)
model.summary()
print('Done Training')
    
###Saving Model and Weights###
with open("model.json", "w") as json_file:
    json_file.write(model.to_json())
model.save('model.h5')
print("Saved model to disk")


