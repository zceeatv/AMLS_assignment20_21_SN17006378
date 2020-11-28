import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os
import pandas as pd
from os.path import dirname, abspath
import numpy as np
import lab2_landmarks as l2
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, Conv2D
#from tensorflow.keras.layers.convolutional import
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.metrics import classification_report,accuracy_score
training_size = 3000

import csv


def get_data():

    X, y = l2.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y

tr_X, tr_Y, te_X, te_Y= get_data()
class_num = te_Y.shape[1]

model = Sequential()
"""
# Convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=tr_X.shape[1:], activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dropout(0.2))
"""

model.add(Dense(136,input_dim=68*2,  kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#model.add(BatchNormalization())

#model.add(Dense(34, kernel_constraint=maxnorm(3)))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))#

model.add(BatchNormalization())
model.add(Dense(64 , kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#model.add(Dense(8, kernel_constraint=maxnorm(3)))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))

model.add(Dense(2))  #Final layer has same number of neurons as classes
model.add(Activation('sigmoid'))

epochs = 30
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(tr_X.reshape((training_size, 68*2)), tr_Y, validation_data=(te_X.reshape((len(te_Y), 68*2)), te_Y), epochs=epochs, batch_size = 10)

scores = model.evaluate(te_X.reshape((len(te_Y), 68*2)), te_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

