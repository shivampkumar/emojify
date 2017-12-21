# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:17:48 2017

@author: shivam
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

cnn= Sequential()
cnn.add(Convolution2D(32,3,3,input_shape=(48,48,1), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(3, 3)))

cnn.add(Flatten())

cnn.add(Dense(output_dim=128, activation='relu'))
cnn.add(Dense(output_dim=6, activation='softmax'))

cnn.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '../dataset/TrainFinal',
        target_size=(48, 48),
        batch_size=50, #check
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../dataset/TestFinal',
        target_size=(48, 48),
        batch_size=30,
        class_mode='categorical')

cnn.fit_generator(
        train_generator,
        steps_per_epoch=3000,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=1200)

model_json = cnn.to_json()

# save model to JSON file
with open("modelgenerated.json", "w") as json_file:
    json_file.write(model_json)


# serialize weights to HDF5
cnn.save_weights("modegenerated.h5")
print("Saved model to disk")






