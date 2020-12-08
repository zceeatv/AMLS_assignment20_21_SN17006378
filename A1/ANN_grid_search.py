from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, Conv2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import landmark_predictor as lp
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
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

def create_model(neurons=1):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation="relu", padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())


    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(neurons, kernel_constraint=maxnorm(3), kernel_initializer='lecun_uniform'))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(neurons/2, kernel_constraint=maxnorm(3),kernel_initializer='lecun_uniform'))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(class_num,kernel_initializer='lecun_uniform'))  #Final layer has same number of neurons as classes
    model.add(Activation('softmax'))

    optimizer = optimizers.Nadam()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    es_callback = EarlyStopping(monitor='val_loss', patience=10)

    return model



# Optimising epoch and batch size
model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [40, 60, 80]
epochs = [10, 20, 30, 40]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(tr_X, tr_Y)

"""
# Optimising Training Optimisation Algorithm
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=60, verbose=0)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(tr_X, tr_Y)


# Optimising Network Weight Initialisation
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=0)
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(tr_X, tr_Y)


# Optimising Neuron Activation Function
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=0)
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(tr_X, tr_Y)


# Optimising Dropout Regularisation
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=0)
weight_constraint = [2, 3, 4]
dropout_rate = [0.1, 0.2, 0.3]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(tr_X, tr_Y)


#Optimising Number of Neurons in Hidden Layer
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=0)
neurons = [256, 128, 64, 32, 16]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(tr_X, tr_Y)
"""
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Model evaluation
#scores = model.evaluate(te_X, te_Y, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
