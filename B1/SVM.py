from B1 import preprocess_data as lp
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
from keras.utils import np_utils


def get_data():
    extract_feature = 1
    X, Y = lp.preprocess(extract_feature)
    dataset_size = X.shape[0]
    training_size = int(dataset_size * 0.7)
    validation_size = training_size + int(dataset_size * 0.1)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    va_X = X[training_size:validation_size]
    va_Y = Y[training_size:validation_size]
    te_X = X[validation_size:]
    te_Y = Y[validation_size:]

    return tr_X, tr_Y, va_X, va_Y, te_X, te_Y


def get_data_import(X, Y):

    dataset_size = X.shape[0]
    training_size = int(dataset_size * 0.7)
    validation_size = training_size + int(dataset_size * 0.10)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    va_X = X[training_size:validation_size]
    va_Y = Y[training_size:validation_size]
    te_X = X[validation_size:]
    te_Y = Y[validation_size:]

    return tr_X, tr_Y, va_X, va_Y, te_X, te_Y


def execute():
    tr_X, tr_Y, va_X, va_Y, te_X, te_Y= get_data()
    """
    #For import preprocessed

    X = np.loadtxt('features.txt')
    X = X.reshape(X.shape[0], X.shape[1] // 2, 2)
    y = np.loadtxt('labels.txt')
    tr_X, tr_Y, va_X, va_Y, te_X, te_Y = get_data_import(X, y)
    """

    tr_Y = np_utils.to_categorical(tr_Y)
    va_Y = np_utils.to_categorical(va_Y)
    te_Y = np_utils.to_categorical(te_Y)

    classifier = svm.SVC(kernel='linear')

    classifier.fit(tr_X.reshape((tr_X.shape[0], 68*2)), list(zip(*tr_Y))[0])
    pred = classifier.predict(tr_X.reshape((tr_X.shape[0], 68 * 2)))
    train_acc = accuracy_score(list(zip(*tr_Y))[0], pred)
    print("Training Accuracy:", train_acc)
    pred = classifier.predict(va_X.reshape((len(va_Y), 68*2)))
    test_acc = accuracy_score(list(zip(*te_Y))[0], pred)
    print("Testing Accuracy:", test_acc)

    return train_acc, test_acc

