from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, Conv2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import landmark_predictor as lp
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""

training_size = 4000
testing = False

def get_data():
    extract_features = 0
    X, Y = lp.preprocess(extract_features)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y


# loading in the data
tr_X, tr_Y, te_X, te_Y = get_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
tr_X = tr_X.astype('float32')
te_X = te_X.astype('float32')
tr_X = tr_X / 255.0
te_X = te_X / 255.0

# reshape to include 1 for grayscale colours
tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1], tr_X.shape[2], 1)
te_X = te_X.reshape(te_X.shape[0], te_X.shape[1], te_X.shape[2], 1)

# one hot encode outputs
tr_Y = np_utils.to_categorical(tr_Y)
te_Y = np_utils.to_categorical(te_Y)
class_num = te_Y.shape[1]
input_shape = (tr_X.shape[1], tr_X.shape[2], 1)

if not testing:
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())


    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(64, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(32, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    """
    model.add(Dense(64, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    """

    model.add(Dense(class_num))  #Final layer has same number of neurons as classes
    model.add(Activation('softmax'))

    epochs = 50
    batch_size = 64
    optimizer = optimizers.Nadam()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    es_callback = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(tr_X, tr_Y, validation_data=(te_X, te_Y), epochs=epochs, batch_size=batch_size)
    model.save("A1_NN_Model")
    print("Saved Neural Network Model")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
else:
    print("Loaded Neural Network Model")
    model = load_model("A1_NN_Model")

# Model evaluation
scores = model.evaluate(te_X, te_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
