import sys
import A1
from A1 import ANN as A1_NN
from A2 import ANN as A2_NN
from B1 import ANN as B1_NN
from B2 import ANN as B2_NN

# ======================================================================================================================
# Data preprocessing
#data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
testing = False
A1_NN.execute(testing)
A2_NN.execute(testing)

B1_NN.execute(testing)
B2_NN.execute(testing)
