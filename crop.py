#!/usr/bin/env python3

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
import cv2
import numpy as np

import glob
import os

from pprint import pprint as pp

from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive

path_in = 'in/*'
path_out = 'out'
window_name = 'crop'
size_max_image = 500
debug_mode = True


def get_image_width_height(image):
    image_width = image.shape[1]  # current image's width
    image_height = image.shape[0]  # current image's height
    return image_width, image_height


def calculate_scaled_dimension(scale, image):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    ratio_of_new_with_to_old = scale / image_width
    dimension = (scale, int(image_height * ratio_of_new_with_to_old))
    return dimension


def rotate_image(image, degree=180):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    center = (image_width / 2, image_height / 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    image_rotated = cv2.warpAffine(image, M, (image_width, image_height))
    return image_rotated


def scale_image(image, size):
    image_resized_scaled = cv2.resize(
        image,
        calculate_scaled_dimension(
            size,
            image
        ),
        interpolation=cv2.INTER_AREA
    )
    return image_resized_scaled

def detect_box(image):
    # convert the image to grayscale, .... 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (debug_mode):  show_image(gray, window_name)

    # blur it, and ...
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if (debug_mode): show_image(gray, window_name)

    # find edges in the image
    edges = cv2.Canny(gray, 75, 200)
    if (debug_mode): show_image(edges, window_name)

    # ind the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (_, contours, _) = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    if (debug_mode):
         cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
         show_image(image, window_name)

    # loop over the contours
    screenCnt = []
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if (debug_mode):
         pp(screenCnt)
         cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 2)
         show_image(image, window_name)

    return image, screenCnt


def show_image(image, window_name, waitForKey=True):
    # Show image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    image_width, image_height = get_image_width_height(image)
    cv2.resizeWindow(window_name, image_width, image_height)

    if waitForKey:
        # Wait before closing
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def cut_of_top(image, pixel):
    image_width, image_height = get_image_width_height(image)

    # startY, endY, startX, endX coordinates
    new_y = 0+pixel
    image = image[new_y:image_height, 0:image_width]
    return image

def cut_of_bottom(image, pixel):
    image_width, image_height = get_image_width_height(image)

    # startY, endY, startX, endX coordinates
    new_height = image_height-pixel
    image = image[0:new_height, 0:image_width]
    return image


for file_iterator in glob.iglob(path_in):
    image = cv2.imread(file_iterator)
    image = rotate_image(image)
    image = cut_of_bottom(image, 1000)

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    if (debug_mode): show_image(image, window_name)
    image, screenCnt = detect_box(image)


    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    if (debug_mode):
        show_image(imutils.resize(warped, height = 650), "Warped")

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    #gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #gray = threshold_adaptive(gray, 251, offset = 10)
    #gray = gray.astype("uint8") * 255

    #if (debug_mode):
    #    show_image(imutils.resize(orig, height = 650), window_name, False)
    #    show_image(imutils.resize(gray, height = 650), "Warped")

    # Create out path
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Build output file path
    file_name_ext = os.path.basename(file_iterator)
    file_name, file_extension = os.path.splitext(file_name_ext)
    file_path = os.path.join(path_out, file_name + '.cropped' + file_extension)

    # Write out file
    cv2.imwrite(file_path, image)

    print("Transform file {} to {}".format(
        file_iterator,
        file_path
    ))
