# AMLS Assignment 20/21 SN17006378

### Organisation
- A1, A2, B1 and B2 folders contain the machine learning code files for each task
- Import your cartoon_set, celeba, cartoon_set_test, celeba_set folders into 'Datasets'
- main.py runs the different machine learning approaches found in the task folders and outputs a table of accuracy score results for each

#### main.py
- Script that will run all the machine learning models for each task
- The user can choose to either load up pretrained neural networks and make predictions on a test set or can train the neural network
from scratch. Make changes to testing variable where necassary:
    ```python
      testing = False # For training or True for testing
    ```
- If user chose to test, the script will print out accuracy scores from predictions on the test data
- if user chose to train, the script will print out accuracy scores from predications on the validation data from the final epoch
as well as the accuracy scores from predications on the testing data
#### A1
##### ANN.py
- Contains the finalised convolutional neural network architecture and training process
- Running execute(testing) will load in datasets, preprocess the data and then either trains the 
    neural network model or loads the pretrained model and performs an accuracy score calculation based 
    on the testing data. 
    ```python
      execute(False) #for training or True for testing
    ```
- During preprocessing the images are grayscale converted and resized to 45 x 55 pixels.

##### ANN_grid_search.py
- Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
- Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
- Requires changing passing variables into create_model for each optimisation option
##### SVM.py
- Contains the SVM algorithm training model and the training process
- Running execute() will load in datasets, preprocess the data and then trains the 
    SVM model, before calculating the accuracy scores from predictions on the test data
- During preprocessing, a facial detection process is carried out based on 68 facial landmarks, returning 68
co-ordinates for each face
##### preprocess_data.py
- Helper library used by all model scripts, used to preprocess data
- Opens faces and labels from selected datasets, change path names where necessary
    ```python
      # PATH TO ALL IMAGES
      basedir = dirname(dirname(abspath(__file__)))
      labels_filename = os.path.join(basedir, 'Datasets')
      labels_filename = os.path.join(labels_filename, 'celeba')   # Change to 'celeba_test' if desired
      labels_filename = os.path.join(labels_filename, 'labels.csv')
    
      images_dir = os.path.join(basedir, 'Datasets')
      images_dir = os.path.join(images_dir, 'celeba') # Change to 'celeba_test' if desired
      images_dir = os.path.join(images_dir, 'img')
    ```
- extract_feature from preprocess() determines whether, facial detection will be carried out 
    on the principle of the 68 facial landmark detection and returns 68 co-ordinate positions for each face
    or will simply convert the images to grayscale and resize them to 45 x 55 pixels


#### A2
##### ANN.py
- Contains the finalised convolutional neural network architecture and training process
- Running execute(testing) will load in datasets, preprocessing the data and then either trains the 
    neural network model or loads the pretrained model and performs an accuracy score calculation based 
    on the testing data. 
    ```python
      execute(False) # For training or True for testing
    ```
- User can determine which of the following two preprocessing configurations they would like to carry out on the dataset.
    Make changes in execute() where necessary:
    ```python
      # Setting configurations:
      crop_mouth = False # Grayscale images and resize to 45 x 55 pixels
      crop_mouth = True # Crop image to mouth section, convert to grayscale and resize to 30 x 60 pixels
  ```
- Cropping the images to just the mouth provides better performance 
##### ANN_grid_search.py
- Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
- Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
- Requires changing passing variables into create_model for each optimisation option
##### SVM.py
- Contains the SVM algorithm training model and the training process
- Running execute() will load in datasets, preprocess the data and then trains the 
    SVM model, before calculating the accuracy scores from predictions on the test data
- User can determine which of the following two preprocessing configurations they would like to carry out on the dataset.
     ```python
          # Setting configurations:
          crop_mouth = False # Facial landmark detection returning all 68 points
          crop_mouth = True # Facial landmark detection returning 19 points only for the mouth area
    ```
- Cropping the facial landmark positions to only those of the mouth provided better performance
##### preprocess_data.py
- Helper library used by all model scripts, used to preprocess data
- Opens faces and labels from selected datasets, change path names where necessary
    ```python
      # PATH TO ALL IMAGES
      basedir = dirname(dirname(abspath(__file__)))
      labels_filename = os.path.join(basedir, 'Datasets')
      labels_filename = os.path.join(labels_filename, 'celeba')   # Change to 'celeba_test' if desired
      labels_filename = os.path.join(labels_filename, 'labels.csv')
    
      images_dir = os.path.join(basedir, 'Datasets')
      images_dir = os.path.join(images_dir, 'celeba') # Change to 'celeba_test' if desired
      images_dir = os.path.join(images_dir, 'img')
    ```
- extract_feature from preprocess() determines whether, facial detection will be carried out 
    on the principle of the 68 facial landmarks and returns 68 co-ordinate positions for each face
    or will simply convert the images to grayscale and resize them to 45 x 55 pixels
- crop_mouth from preprocess() determines whether the mouth area of the face will be extracted. In the case where 
feature extraction took place, the co-ordinates of the 19 points of the mouth are taken. If there was no feature
extraction, the image of the mouth is extracted.

#### B1
##### ANN.py
- Contains the finalised convolutional neural network architecture and training process
- Running execute(testing) will load in datasets, preprocessing the data and then either trains the 
    neural network model or loads the pretrained model and performs an accuracy score calculation based 
    on the testing data. 
    ```python
         execute(False) # For training or True for testing
    ```
##### ANN_grid_search.py
- Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
- Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
- Requires changing passing variables into create_model for each optimisation option
##### preprocess_data.py
- Helper library used by all model scripts, used to preprocess data
- Opens faces and labels from selected datasets, change path names where necessary
    ```python
      # PATH TO ALL IMAGES
      basedir = dirname(dirname(abspath(__file__)))
      labels_filename = os.path.join(basedir, 'Datasets')
      labels_filename = os.path.join(labels_filename, 'cartoon_set')   # Change to 'cartoon_set_test' if desired
      labels_filename = os.path.join(labels_filename, 'labels.csv')
    
      images_dir = os.path.join(basedir, 'Datasets')
      images_dir = os.path.join(images_dir, 'cartoon_set') # Change to 'cartoon_set_test' if desired
      images_dir = os.path.join(images_dir, 'img')
    ```
- Will simply convert the images to grayscale and resize them to 50 x 50 pixels along with the corresponding labels
#### B2
##### ANN.py
- Contains the finalised convolutional neural network architecture and training process
- Running execute(testing) will load in datasets, preprocess the data and then either trains the 
    neural network model or loads the pretrained model and performs an accuracy score calculation based 
    on the testing data. 
    ```python
      execute(False) # For training or True for testing
    ```
- User can determine which of the following two preprocessing configurations they would like to carry out on the dataset.
    Make changes in execute() where necessary:
    ```python
      # Setting configurations:
      crop = False # Resize to 50 x 50 pixels
      crop = True # Crop image to left eye, and resize to 30 x 50 pixels
  ```
- Croping the faces to just the eye provides better performance
##### ANN_grid_search.py
- Standalone program used to carry out grid search to optimise the hyperparameters of the Neural Network
- Hyperparameters include epoch and batch size, training optimisation algorithm, network weight initialisation
    neuron activation function, dropout regularisation, number of neurons in hidden layer
- Requires changing passing variables into create_model for each optimisation option
##### preprocess_data.py
- Helper library used by all model scripts, used to preprocess data
- Opens faces and labels from selected datasets, change path names where necessary
    ```python
      # PATH TO ALL IMAGES
      basedir = dirname(dirname(abspath(__file__)))
      labels_filename = os.path.join(basedir, 'Datasets')
      labels_filename = os.path.join(labels_filename, 'cartoon_set')   # Change to 'cartoon_set_test' if desired
      labels_filename = os.path.join(labels_filename, 'labels.csv')
    
      images_dir = os.path.join(basedir, 'Datasets')
      images_dir = os.path.join(images_dir, 'cartoon_set') # Change to 'cartoon_set_test' if desired
      images_dir = os.path.join(images_dir, 'img')
    ```
- crop from preprocess() determines whether the image is simply resized to 50 x 50 pixels and returned, or is cropped to the left eye and resized to 30 x 50 pixels

- If the preprocessing is ran for training (testing=False), there is further detection for faces with sunglasses which are removed from 
the returned dataset
- If the preprocessing is ran for testing (testing=True) then these are not omitted

#### System used for training Models
Computational times provided in this READ.me are based off the following system configurations:
 - Intel Core i7-8750H	2.2-4.1GHz	6/12 cores	9 MB cache
 - Nvidia Quadro P1000	640 CUDA 1.49-1.52GHz	4 GB GDDR5
 - Setup of CUDA toolkit allowing Quadro P1000 for use in training tenserflow models
 
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
    

    
