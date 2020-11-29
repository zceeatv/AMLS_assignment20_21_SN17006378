#import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import os
import pandas as pd
from os.path import dirname, abspath
from numpy import savetxt

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = dirname(dirname(abspath(__file__)))
labels_filename = os.path.join(basedir, 'datasets\\cartoon_set\\labels.csv')
images_dir = os.path.join(basedir, 'datasets\\cartoon_set\\img')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def get_features(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('float32')


    # detect faces in the grayscale image
    rects = detector(resized_image, 1)
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
        temp_shape = predictor(resized_image, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')
    resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = resized_image.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(resized_image, 1)
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
        temp_shape = predictor(resized_image, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
    if dlibout is not None:
        eyes = dlibout[36:41]
        # print(np.amin(mouth, axis=0)[0])
        corners = np.array([[np.amin(eyes, axis=0)[0], np.amin(eyes, axis=0)[1]], [np.amax(eyes, axis=0)[0], np.amax(eyes, axis=0)[1]]])
        eyes = resized_image[corners[0][1]-5:corners[1][1]+5, corners[0][0]-5:corners[1][0]]

        eyes = cv2.resize(eyes, (50, 30), interpolation=cv2.INTER_AREA)
        temp = eyes.reshape(50*30,3)
        count = 0
        colour = temp[500]
        for pixel in temp:
            if np.array_equal(pixel,colour):
                count += 1
        if count > 1200:
            return None,resized_image
    return eyes, resized_image


def extract_eyes():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:     dw which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None

    column = [0, 1]
    df = pd.read_csv(labels_filename, delimiter="\t", usecols=column)
    eye_colours = {}
    for index, row in df.iterrows():
        eye_colours[str(index)] = (row['eye_color'])
    print("done")
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        count = 0
        error = []
        for img_path in image_paths:
            file_name=img_path.split('\\')[-1].split('.')[0]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                count += 1
                all_features.append(features)
                all_labels.append(eye_colours[file_name])
                if(count == 10000):
                    break
            else:
                error.append(file_name)


    landmark_features = np.array(all_features)
    arr_reshaped = landmark_features.reshape(landmark_features.shape[0], -1)
    np.savetxt("features.txt", arr_reshaped)

    eye_colours = np.array(all_labels)  # simply converts the -1 into 0, so male=0 and female=1
    np.savetxt("labels.txt", eye_colours)
    return landmark_features, eye_colours

def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None

    column = [0, 2]
    df = pd.read_csv(labels_filename, delimiter="\t", usecols=column)
    face_shapes = {}
    for index, row in df.iterrows():
        face_shapes[str(index)] = (row['face_shape'])
    print("done")
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        count = 0
        error = []
        for img_path in image_paths:
            file_name=img_path.split('\\')[-1].split('.')[0]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = get_features(img)
            if features is not None:
                count += 1
                all_features.append(features)
                all_labels.append(face_shapes[file_name])
                if(count == 5000):
                    break
            else:
                error.append(file_name)

    landmark_features = np.array(all_features)
    face_shapes = (np.array(all_labels) + 1)/2  # simply converts the -1 into 0, so male=0 and female=1

    arr_reshaped = landmark_features.reshape(landmark_features.shape[0], -1)
    np.savetxt("features.txt", arr_reshaped)
    np.savetxt("labels.txt", face_shapes)
    return landmark_features, face_shapes

def preprocess():
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None

    column = [0, 2]
    df = pd.read_csv(labels_filename, delimiter="\t", usecols=column)
    face_shapes = {}
    for index, row in df.iterrows():
        face_shapes[str(index)] = (row['face_shape'])
    print("formatted labels")
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        count = 0
        error = []
        for img_path in image_paths:
            file_name=img_path.split('\\')[-1].split('.')[0]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            resized_image = img.astype('uint8')

            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            gray = gray.astype('uint8')
            gray = cv2.resize(gray, (50, 50), interpolation=cv2.INTER_AREA)
            all_features.append(gray)
            all_labels.append(face_shapes[file_name])
            if(count == 5000):
                break
    print("preprocess completed")
    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1)/2  # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels