import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import os
import pandas as pd
from os.path import dirname, abspath, split
from numpy import savetxt


# PATH TO ALL IMAGES
basedir = dirname(dirname(abspath(__file__)))
labels_filename = os.path.join(basedir, 'datasets')
labels_filename = os.path.join(labels_filename, 'cartoon_set')
labels_filename = os.path.join(labels_filename, 'labels.csv')

images_dir = os.path.join(basedir, 'datasets')
images_dir = os.path.join(images_dir, 'cartoon_set')
images_dir = os.path.join(images_dir, 'img')

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


def get_features(image):
    """
    This function loads the image, detects the landmarks of the face
    :return:
        dlibout:  an array containing 68 landmark points
        resized_image: an array containing processed images
    """
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


def crop_eye(image, testing):
    """
    This function loads the image, detects the landmarks of the face, and crops the eye for each image
    :return:
        dlibout:  an array cropped eyes for each face
        resized_image: an array containing processed images
    """
    # load the input image, resize it, and convert it to grayscale
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
        if not testing:
            exclude = identify_sunglasses(eyes)
            if exclude:
                return None, resized_image
            else:
                return eyes, resized_image
        else:
            return eyes, resized_image


def identify_sunglasses(eyes):
    reshape = eyes.reshape(50*30, 3)
    count = 0
    colour = reshape[500]
    for pixel in reshape:
        if np.array_equal(pixel, colour):
            count += 1
    if count > 1000:
        return True
    else:
        return False


def preprocess(crop, testing):
    """
    This function loads all the images in the folder 'dataset/cartoon_set'. Converts them to grayscale
    and resizes the images to smaller sizes for faster processing of the neural network
    :param crop: True or False for if the faces are to be cropped to just the eye
    :param testing: true or False for if neglecting sunglasses will be used or not
    :return:
            faces:  an array of resized grayscaled images
            face_shapes:      an array containing the face shape labels
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    column = [0, 1]     # Takes the columns from the labels file that correspond to the index and the face shape label
    df = pd.read_csv(labels_filename, delimiter="\t", usecols=column)
    eye_colours = {}
    for index, row in df.iterrows():  # Goes through each row and extracts the index and the eye colour label
        eye_colours[str(index)] = (row['eye_color'])   # Creates a dictionary entry with key = index and item= eye colour label
    print("Begin Preprocessing faces")
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        count = 0
        error = []
        for img_path in image_paths:
            file_name = split(img_path)[1].split('.')[0]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))

            if crop:
                features, _ = crop_eye(img, testing)   # Get features
            else:
                features = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                features = features.astype('uint8')
                features = cv2.resize(features, (50, 50), interpolation=cv2.INTER_AREA)
            if features is not None:
                all_features.append(features)
                all_labels.append(eye_colours[file_name])
                if(count == 10000):
                    break
            else:
                error.append(file_name) # If the dlib facial predictor could not detect facial features, add to error list for future reference
    print("Finished preprocessing faces")
    faces = np.array(all_features)
    eye_colours = np.array(all_labels)

    """For Saving to text files
    arr_reshaped = landmark_features.reshape(landmark_features.shape[0], -1)
    np.savetxt("features.txt", arr_reshaped)
    np.savetxt("labels.txt", eye_colours)
    """

    return faces, eye_colours

