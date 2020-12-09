# AMLS Assignment 20/21 SN17006378

### Organisation
- A1, A2, B1 and B2 folders contain the machine learning code files for each task
- Import your cartoon_set, celeba, cartoon_set_test, celeba_set folders into 'Datasets'
- main.py runs the different machine learning approaches found in the task folders and outputs a table of accuracy score results for each

### Machine Learning Approaches

#### A1
- ANN.py
    - Contains the finalised convolutional neural network architecture and training process
    - Running execute(testing) will load in datasets, preprocessing the data and then either trains the 
    neural network model or loads the pretrained model and performs an accuracy score calculation based 
    on the testing data. 
    ```python
         execute(False) #for training
         execute(True) #for testing
    ```
    - extract_features sets whether during preprocessing, a facial detector detects 68 positions of the face
    or if the images are grayscale converted and resized to 45 x 55 pixels. Make changes in execute() where neccasary:
    ```python
      def execute(testing):
          extract_features = True # True = facial detection, False = grayscale and resize
          tr_X, tr_Y, va_X, va_Y, te_X, te_Y = get_data(extract_features)
    ```
- ANN_grid_search.py
    - Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
    - Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
    - Requires changing passing variables into create_model for each optimisation option
- preprocess_data.py
    - Helper library used by all model scripts, used to preprocess data
    - Opens faces and labels from selected datasets
    ```python
      # PATH TO ALL IMAGES
      basedir = dirname(dirname(abspath(__file__)))
      labels_filename = os.path.join(basedir, 'Datasets')
      labels_filename = os.path.join(labels_filename, 'celeba')   # Change to celeba_test if desired
      labels_filename = os.path.join(labels_filename, 'labels.csv')
    
      images_dir = os.path.join(basedir, 'Datasets')
      images_dir = os.path.join(images_dir, 'celeba') # Change to celeba_test if desired
      images_dir = os.path.join(images_dir, 'img')
    ```
    - Depending on passed variable extract_feature into preprocess(), facial detection will be carried out 
    on the principle of the 68 facial landmark detection and returns 68 co-ordinate positions for each face
    - Or will simply convert the images to grayscale and resize them to 45 x 55 pixels
- SVM.py
#### A2
- ANN.py
    - Contains the finalised convolutional neural network architecture and training process
- ANN_grid_search.py
    - Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
    - Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
- preprocess_data.py
- SVM.py

#### B1
- ANN.py
    - Contains the finalised convolutional neural network architecture and training process
- ANN_grid_search.py
    - Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
    - Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
- preprocess_data.py

#### B2
- ANN.py
    - Contains the finalised convolutional neural network architecture and training process
- ANN_grid_search.py
    - Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
    - Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
- preprocess_data.py

### Packages requirements:
- Numpy 1.19.2
- Opencv 3.4.2
- Pandas 1.1.3
- Tensorflow 2.1.0
- Keras 2.3.1
- Scikit-learn 0.23.2
- Dlib 19.21.0

#### Installing Packages

- Create Anaconda environment running python 3.7.9
- Install the Numpy, OpenCV, Pandas, Tensorflow, Keras, Scikit-learn packages from the Anaconda Navigator
- For Windows
    - Ensure that Visual studio is installed on your desktop
    - Run "pip install cmake" and "pip install dlib"
- For other operating systems
    - Follow instuctions at https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
    

    
