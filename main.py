
from A1 import ANN as A1_NN
from A1 import SVM as A1_SVM
from A2 import ANN as A2_NN
from A2 import SVM as A2_SVM
from B1 import ANN as B1_NN
from B1 import SVM as B1_SVM
from B2 import ANN as B2_NN

# ======================================================================================================================
# Data preprocessing
#data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================

# Task A1
testing = False
if not testing:
    print("Training model for A1:\n")
    A1_NN_train, A1_NN_test = A1_NN.execute(testing)
    A1_SVM_train, A1_SVM_test = A1_SVM.execute()

    # Task A2
    print("Training model for A2:\n")
    A2_NN_train, A2_NN_test = A2_NN.execute(testing)
    A2_SVM_train, A2_SVM_test = A2_SVM.execute()

    # Task B1
    print("Training model for B1:\n")
    B1_NN_train, B1_NN_test = B1_NN.execute(testing)
    B1_SVM_train, B1_SVM_test = B1_SVM.execute()

    # Task B2
    print("Training model for B2:\n\n")
    B2_NN_train, B2_NN_test = B2_NN.execute(testing)

    print("Model   Train Acc   Test Acc")
    print('TA1_NN:  {:.2f},     {:.2f};\nTA1_SVM:   {:.2f},     {:.2f};\nTA2_NN:    {:.2f},     {:.2f};\nTA2_SVM:   {:.2f},     {:.2f};\nTB1_NN:    {:.2f},     {:.2f};\nTB1_SVM:   {:.2f},     {:.2f};\nTB2_NN:    {:.2f},     {:.2f};\n'.format(
        A1_NN_train, A1_NN_test,
        A1_SVM_train, A1_SVM_test,
        A2_NN_train, A2_NN_test,
        A2_SVM_train, A2_SVM_test,
        B1_NN_train, B1_NN_test,
        B1_SVM_train, B1_SVM_test,
        B2_NN_train, B2_NN_test))
    """
    A1_NN_train, A1_NN_test,
    A1_SVM_train, A1_SVM_test,
    A2_NN_train, A2_NN_test,
    A2_SVM_train, A2_SVM_test,
    B1_NN_train, B1_NN_test,
    B1_SVM_train, B1_SVM_test,
    B2_NN_train, B2_NN_test))
    
    """

else:
    # Task A1
    print("Training model for A1:\n")
    A1_NN_test = A1_NN.execute(testing)
    #A1_SVM_train, A1_SVM_test = A1_SVM.execute()

    # Task A2
    print("Training model for A2:\n")
    A2_NN_test = A2_NN.execute(testing)
    #A2_SVM_train, A2_SVM_test = A2_SVM.execute()

    # Task B1
    print("Training model for B1:\n")
    B1_NN_test = B1_NN.execute(testing)
    #B1_SVM_train, B1_SVM_test = B1_SVM.execute()

    # Task B2
    print("Training model for B2:\n")
    B2_NN_test = B2_NN.execute(testing)

    print("Model Test_Acc")
    print('TA1_NN:  {:.2f};\nTA2_NN:    {:.2f};\nTB1_NN:    {:.2f};\nTB2_NN:    {:.2f};\n'.format(A1_NN_test, A2_NN_test, B1_NN_test, B2_NN_test))



