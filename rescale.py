import cv2 as cv

img = cv.imread('Images/First.jpg')
cv.imshow('First', img)


def rescaleFrame(frame, scale=0.125):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_image = rescaleFrame(img)
cv.imshow('Image', resized_image)

cv.waitKey(0)
