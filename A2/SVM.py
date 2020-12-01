import landmark_predictor as lp
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm


def get_data(crop):
    extract_feature = 1
    X, y = lp.preprocess(extract_feature, crop)
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y


def get_data_import(X, y):

    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:training_size]
    tr_Y = Y[:training_size]
    te_X = X[training_size:]
    te_Y = Y[training_size:]

    return tr_X, tr_Y, te_X, te_Y


# sklearn functions implementation
def img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))

    print(pred)
    return pred


training_size = 3000
crop = 0
tr_X, tr_Y, te_X, te_Y= get_data(crop)
"""
#For import preprocessed

X = np.loadtxt('features-1.txt')
X = X.reshape(X.shape[0], X.shape[1] // 2, 2)
y = np.loadtxt('labels.txt')
tr_X, tr_Y, te_X, te_Y= get_data_import(X, y)

"""
if not crop:
    pred=img_SVM(tr_X.reshape((training_size, 68*2)), list(zip(*tr_Y))[0], te_X.reshape((len(te_Y), 68*2)), list(zip(*te_Y))[0])
else:
    pred=img_SVM(tr_X.reshape((training_size, 19*2)), list(zip(*tr_Y))[0], te_X.reshape((len(te_Y), 19*2)), list(zip(*te_Y))[0])
