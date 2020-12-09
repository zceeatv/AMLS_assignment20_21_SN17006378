from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, Conv2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from A1 import preprocess_data as lp
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""


def get_data(extract_features):
    X, Y = lp.preprocess(extract_features)
    dataset_size = X.shape[0]
    training_size = int(dataset_size * 0.8)
    validation_size = training_size + int(dataset_size * 0.1)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    va_X = X[training_size:validation_size]
    va_Y = Y[training_size:validation_size]
    te_X = X[validation_size:]
    te_Y = Y[validation_size:]

    return tr_X, tr_Y, va_X, va_Y, te_X, te_Y


def execute(testing):
    # Global variables
    extract_features = True

    # loading in the data
    tr_X, tr_Y, va_X, va_Y, te_X, te_Y = get_data(extract_features)

    # normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
    tr_X = tr_X.astype('float32')
    va_X = va_X.astype('float32')
    te_X = te_X.astype('float32')
    tr_X = tr_X / 255.0
    te_X = te_X / 255.0
    va_X = va_X / 255.0


    # reshape to include 1 for grayscale colours
    tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1], tr_X.shape[2], 1)
    va_X = va_X.reshape(va_X.shape[0], va_X.shape[1], va_X.shape[2], 1)
    te_X = te_X.reshape(te_X.shape[0], te_X.shape[1], te_X.shape[2], 1)

    # one hot encode outputs
    tr_Y = np_utils.to_categorical(tr_Y)
    va_Y = np_utils.to_categorical(va_Y)
    te_Y = np_utils.to_categorical(te_Y)
    class_num = te_Y.shape[1]
    input_shape = (tr_X.shape[1], tr_X.shape[2], 1)

    if not testing:
        model = Sequential()

        # Convolutional layers
        model.add(Conv2D(100, (4, 4), input_shape=input_shape, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Conv2D(200, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.3))
        model.add(BatchNormalization())
        #model.add(Conv2D(200, (2, 2), activation='relu', padding='same'))

        model.add(Flatten())
        #model.add(Dropout(0.3))

        model.add(Dense(128, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Dense(64, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Dense(class_num))  #Final layer has same number of neurons as classes
        model.add(Activation('softmax'))

        epochs = 5
        batch_size = 64
        optimizer = optimizers.Adam(lr=0.0001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        es_callback = EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(tr_X, tr_Y, validation_data=(va_X, va_Y), epochs=epochs, batch_size=batch_size)
        #model.save("A1_NN_Model")
        #print("Saved Neural Network Model")
        plt.plot(history.history['loss'],marker='x')
        plt.plot(history.history['val_loss'], marker='x')
        plt.title('Learning Rate Curve for CNN')
        plt.ylabel('Cost')
        plt.xlabel('Number of Epochs')
        plt.legend(['train', 'test'], loc='upper left')
        plt.rcParams.update({'font.size': 22})
        plt.show()
    else:
        print("Loaded Neural Network Model")
        model = load_model("A1_NN_Model")

    # Model evaluation
    scores = model.evaluate(te_X, te_Y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

execute(False)

