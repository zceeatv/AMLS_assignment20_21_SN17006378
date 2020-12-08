
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, Conv2D
from keras.constraints import maxnorm
from keras.utils import np_utils
#import landmark_predictor as lp
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import os
import pandas as pd
from os.path import dirname, abspath, split
from numpy import savetxt

# PATH TO ALL IMAGES
basedir = dirname(dirname(abspath(__file__)))
labels_filename = os.path.join(basedir, 'datasets')
labels_filename = os.path.join(labels_filename, 'cartoon_set')
labels_filename = os.path.join(labels_filename, 'labels.csv')

images_dir = os.path.join(basedir, 'datasets')
images_dir = os.path.join(images_dir, 'cartoon_set')
images_dir = os.path.join(images_dir, 'img')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""

training_size = 7000
testing = False

def get_data():
    X, Y = preprocess()
    dataset_size = X.shape[0]
    training_size = int(dataset_size * 0.7)
    validation_size = training_size + int(dataset_size * 0.15)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    va_X = X[training_size:validation_size]
    va_Y = Y[training_size:validation_size]
    te_X = X[validation_size:]
    te_Y = Y[validation_size:]

    return tr_X, tr_Y, va_X, va_Y, te_X, te_Y


def get_data_import(X,Y):
    dataset_size = X.shape[0]
    training_size = int(dataset_size * 0.7)
    validation_size = training_size + int(dataset_size * 0.15)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    va_X = X[training_size:validation_size]
    va_Y = Y[training_size:validation_size]
    te_X = X[validation_size:]
    te_Y = Y[validation_size:]

    return tr_X, tr_Y, va_X, va_Y, te_X, te_Y

def preprocess():
    """
    This function loads all the images in the folder 'dataset/cartoon_set'. Converts them to grayscale
    and resizes the images to smaller sizes for faster processing of the neural network
    :return:
        faces:  an array of resized grayscaled images
        face_shapes:      an array containing the face shape labels
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    column = [0, 2]     # Takes the columns from the labels file that correspond to the index and the face shape label
    df = pd.read_csv(labels_filename, delimiter="\t", usecols=column)
    face_shapes = {}
    for index, row in df.iterrows():  # Goes through each row and extracts the index and the face shape label
        face_shapes[str(index)] = (row['face_shape'])   # Creates a dictionary entry with key = index and item= face shape label
    print("Begin Preprocessing faces")
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        count = 0
        error = []
        for img_path in image_paths:
            file_name = split(img_path)[1].split('.')[0]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            resized_image = img.astype('uint8')

            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            gray = gray.astype('uint8')
            gray = cv2.resize(gray, (50, 50), interpolation=cv2.INTER_AREA)
            all_features.append(gray)
            all_labels.append(face_shapes[file_name])
            if(count == 10000):
                break
    print("Finished preprocessing faces")
    faces = np.array(all_features)
    face_shapes = np.array(all_labels)

    """For Saving to text files
    arr_reshaped = landmark_features.reshape(landmark_features.shape[0], -1)
    np.savetxt("features.txt", arr_reshaped)
    np.savetxt("labels.txt", eye_colours)
    """

    return faces, face_shapes

def execute(testing):
    """
    # loading in the data
    X = np.loadtxt('features.txt')
    X = X.reshape(X.shape[0], X.shape[1] // 2, 2)
    y = np.loadtxt('labels.txt')
    tr_X, tr_Y, te_X, te_Y= get_data_import(X,y)
    """

    # loading in the data
    tr_X, tr_Y, va_X, va_Y, te_X, te_Y = get_data()

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
        model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='linear', padding='same'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.2))

        model.add(Dense(256, kernel_constraint=maxnorm(4)))
        model.add(Activation('linear'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(128, kernel_constraint=maxnorm(4)))
        model.add(Activation('linear'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        """
        model.add(Dense(64, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        """

        model.add(Dense(class_num))   #Final layer has same number of neurons as classes
        model.add(Activation('softmax'))

        epochs = 10
        batch_size = 64
        optimizer = 'adam'

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        #es_callback = EarlyStopping(monitor='val_loss', patience=3)
        # , callbacks=[es_callback]
        history = model.fit(tr_X, tr_Y, validation_data=(te_X, te_Y), epochs=epochs, batch_size=batch_size)
        model.save("B1_NN_Model")
        print("Saved Neural Network Model")
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
        model = load_model("B1_NN_Model")

    # Model evaluation
    scores = model.evaluate(te_X, te_Y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
