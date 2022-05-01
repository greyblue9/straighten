import cv2 as cv
import numpy as np
from sklearn.exceptions import DataDimensionalityWarning

img = cv.imread('Images/First.jpg')
#cv.imshow('First', img)


def rescaleFrame(frame, scale=0.125):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_image = rescaleFrame(img)
cv.imshow('Image', resized_image)

# Translation


def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
#  x --> Right
#  y --> Down


translated = translate(resized_image, -100, 100)
cv.imshow('Translated', translated)

# Rotation


def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(resized_image, rotMat, dimensions)


rotated = rotate(resized_image, 45)
cv.imshow('Rotated', rotated)


# Resize

resize = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resize)

# Flipping
flip = cv.flip(resized_image, 1)
cv.imshow('Flip', flip)

#

# 0 - vertical 1 - horizontal -1 - both

cv.waitKey(0)
