import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, Conv2D
#from tensorflow.keras.layers.convolutional import
from keras.constraints import maxnorm
from keras.utils import np_utils
import landmark_predictor as lp
from tensorflow.keras.callbacks import EarlyStopping

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""

training_size = 4000
def get_data():
    X, Y = lp.preprocess()
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y

# loading in the data
tr_X, tr_Y, te_X, te_Y= get_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255

tr_X = tr_X.astype('float32')
te_X = te_X.astype('float32')
tr_X = tr_X / 255.0
te_X = te_X / 255.0
tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1], tr_X.shape[2], 1)
te_X = te_X.reshape(te_X.shape[0], te_X.shape[1], te_X.shape[2], 1)

# one hot encode outputs
tr_Y = np_utils.to_categorical(tr_Y)
te_Y = np_utils.to_categorical(te_Y)
class_num = te_Y.shape[1]
input_shape = (tr_X.shape[1], tr_X.shape[2], 1)

model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

"""
model.add(Dense(64, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
"""
model.add(Dense(class_num))  #Final layer has same number of neurons as classes
model.add(Activation('softmax'))

epochs = 40
batch_size = 10
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#es_callback = EarlyStopping(monitor='val_loss', patience=5)

model.fit(tr_X, tr_Y, validation_data=(te_X, te_Y), epochs=epochs, batch_size=batch_size)

# Model evaluation
scores = model.evaluate(te_X, te_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))