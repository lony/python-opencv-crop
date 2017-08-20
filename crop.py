#!/usr/bin/env python3

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
import cv2
import numpy as np

window_size = 800
window_name = 'crop'

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
image = cv2.imread('image.jpg')

def get_image_width_height(image):
    image_width = image.shape[1]	# current image's width
    image_height = image.shape[0]	# current image's height 
    return image_width, image_height

def calculate_scaled_dimension(scale, image):
    # http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
    image_width, image_height = get_image_width_height(image)
    ratio_of_new_with_to_old = scale / image_width
    dimension = (scale, int(image_height * ratio_of_new_with_to_old))
    return dimension

# Scale image to size
image_resized_scaled = cv2.resize(
    image,
    calculate_scaled_dimension(
        window_size,
        image
    ),
    interpolation = cv2.INTER_AREA
)

# Show image
cv2.imshow(window_name, image_resized_scaled)
image_width, image_height = get_image_width_height(image_resized_scaled)
cv2.resizeWindow(window_name, image_width, image_height)

# Wait before closing
cv2.waitKey(0)
cv2.destroyAllWindows()