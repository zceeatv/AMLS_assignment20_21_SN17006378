
from A1 import ANN as A1_NN
from A2 import ANN as A2_NN
from B1 import ANN as B1_NN
from B2 import ANN as B2_NN
#from A1 import landmark_predictor as lp
#from A2 import landmark_predictor
#from B1 import landmark_predictor

#from B2 import landmark_predictor

# ======================================================================================================================
# Data preprocessing
#data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
testing = False
A1_NN.execute(testing)

# Task A2
A2_NN.execute(testing)

# Task B1
B1_NN.execute(testing)

# Task B2
B2_NN.execute(testing)

"""
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
"""
