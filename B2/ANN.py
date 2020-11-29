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

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""
training_size = 5000
def get_data():
    X, Y = lp.extract_eyes()
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y

def get_data_import(X,Y):
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y
# loading in the data


X = np.loadtxt('features.txt')
X = X.reshape(X.shape[0], 30, 50, 3)
y = np.loadtxt('labels.txt')
tr_X, tr_Y, te_X, te_Y= get_data_import(X,y)


#tr_X, tr_Y, te_X, te_Y= get_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255

tr_X = tr_X.astype('float32')
te_X = te_X.astype('float32')
tr_X = tr_X / 255.0
te_X = te_X / 255.0
tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1], tr_X.shape[2], 3)
te_X = te_X.reshape(te_X.shape[0], te_X.shape[1], te_X.shape[2], 3)

# one hot encode outputs
tr_Y = np_utils.to_categorical(tr_Y)
te_Y = np_utils.to_categorical(te_Y)
class_num = te_Y.shape[1]
input_shape = (tr_X.shape[1], tr_X.shape[2], 3)

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
batch_size = 64
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
es_callback = EarlyStopping(monitor='val_loss', patience=3)

model.fit(tr_X, tr_Y, validation_data=(te_X, te_Y), epochs=epochs, batch_size=batch_size , callbacks=[es_callback])

# Model evaluation
scores = model.evaluate(te_X, te_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))