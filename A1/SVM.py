from A1 import preprocess_data as lp
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
from keras.utils import np_utils


def get_data():

    X, Y = lp.preprocess(1)
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y


def get_data_import(X, Y):
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y


training_size = 3000
def execute():
    """
    For import preprocessed

    X = np.loadtxt('features.txt')
    X = X.reshape(X.shape[0], X.shape[1] // 2, 2)
    y = np.loadtxt('labels.txt')
    tr_X, tr_Y, te_X, te_Y= get_data_import(X,y)
    """

    # Manual preprocess
    tr_X, tr_Y, te_X, te_Y= get_data()
    tr_Y = np_utils.to_categorical(tr_Y)
    te_Y = np_utils.to_categorical(te_Y)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(tr_X.reshape((tr_X.shape[0], 68 * 2)), list(zip(*tr_Y))[0])
    pred = classifier.predict(tr_X.reshape((tr_X.shape[0], 68*2)))
    train_acc = accuracy_score(list(zip(*tr_Y))[0], pred)
    print("Training Accuracy:", train_acc)
    pred = classifier.predict(te_X.reshape((len(te_Y), 68*2)))
    test_acc = accuracy_score(list(zip(*te_Y))[0], pred)
    print("Testing Accuracy:", test_acc)

    return train_acc, test_acc
