import numpy as np
import cv2 as cv

color = cv.imread('Images/Third.jpg', cv.IMREAD_COLOR)


def rescaleFrame(frame, scale=0.125):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_image = rescaleFrame(color)
cv.imshow('Image', resized_image)
#cv.imshow('img', img)
gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)

dst = cv.dilate(dst, None)

resized_image[dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('dst', resized_image)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
