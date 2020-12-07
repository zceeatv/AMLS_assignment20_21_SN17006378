import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import os
import pandas as pd
from os.path import dirname, abspath, split
from numpy import savetxt

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = dirname(dirname(abspath(__file__)))
labels_filename = os.path.join(basedir, 'datasets\\celeba\\labels.csv')
images_dir = os.path.join(basedir, 'datasets\\celeba\\img')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
    """
    Takes the facial landmarks created from the dlib functions and creates a reshaped numpy array
    :return: numpy array with shape (68,2)
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    """
    Takes a bounding predicted by dlib and convert it to the format (x, y, w, h) used with OpenCV
    :return: A list of x, y, w, h values
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def crop_mouth(image, extract_features, crop):  # Get facial feature and then crop to the mouth area
    """
    This function loads the image, detects the landmarks of the face or crops out the mouth of the image
    :param extract_features: True or False for if extracting features of the mouth
    :param crop: True or False for if cropping the mouth from faces
    :return:
    """
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep

    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
    if extract_features and crop:
        dlibout = dlibout[48:67]
    elif extract_features and not crop:
        dlibout = dlibout
    elif not extract_features:
        dlibout = dlibout[48:67]
        corners = np.array([[np.amin(dlibout, axis=0)[0], np.amin(dlibout, axis=0)[1]], [np.amax(dlibout, axis=0)[0], np.amax(dlibout, axis=0)[1]]])
        mouth = gray[corners[0][1]-10:corners[1][1]+10, corners[0][0]-10:corners[1][0]+10]
        dlibout = cv2.resize(mouth, (60, 30), interpolation=cv2.INTER_AREA)

    """
        cv2.imshow("crop", dlibout)
        cv2.waitKey(0)
        """
    return dlibout, resized_image

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert all colours of the image to grayscale
    gray = gray.astype('uint8')
    gray = cv2.resize(gray, (45*2, 55*2), interpolation=cv2.INTER_AREA)  # Decrease the size of the image
    return gray


def preprocess(extract_feature, crop):   # Grayscale and Resize all images
    """
    This function loads all the images in the folder 'dataset/cartoon_set'. Converts them to grayscale
    and resizes the images to smaller sizes for faster processing of the neural network
    :param crop: True or False for if the faces are to be cropped to just the mouth
    :param extract_feature: true or False for if extracting features or to just preprocess faces
    :return:
            faces:  an array resized grayscaled images or features
            smile_labels:  an array containing the smile labels
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None

    column = [0, 3]  # Takes the columns from the labels file that correspond to the index and the smile label
    df = pd.read_csv(labels_filename, delimiter="\t", usecols=column)
    smile_labels = {}
    for index, row in df.iterrows():  # Goes through each row and extracts the index and the smile label
        smile_labels[str(index)] = (row['smiling'])  # Creates a dictionary entry with key = index and item= smile label
    print("Begin processing faces")
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        count = 0
        error = []
        for img_path in image_paths:
            file_name = split(img_path)[1].split('.')[0]    # Get's the number for each image from the file path

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            if not extract_feature and not crop:
                features = process_image(img)
            else:
                features, _ = crop_mouth(img, extract_feature, crop)   # Get features

            if features is not None:
                all_features.append(features)
                all_labels.append(smile_labels[file_name])
                if(count == 5000):
                    break
            else:
                error.append(file_name) # If the dlib facial predictor could not detect facial features, add to error list for future reference
    print("Finished processing faces")
    faces = np.array(all_features)
    smile_labels = (np.array(all_labels) + 1)/2  # simply converts the -1 into 0, so male=0 and female=1
    return faces, smile_labels

    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """