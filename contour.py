import cv2 as cv
from matplotlib.pyplot import contour
import numpy as np

img = cv.imread('Images/Third.jpg')
#cv.imshow('First', img)


def rescaleFrame(frame, scale=0.125):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_image = rescaleFrame(img)
cv.imshow('Image', resized_image)

blank = np.zeros(resized_image.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)

#canny = cv.Canny(gray, 50, 50)
#cv.imshow('Canny', canny)

ret, thresh = cv.threshold(gray, 125, 175, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# contours, hierarchies = cv.findContours(
# canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE

# contours, hierarchies = cv.findContours(
#   canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

contours, hierarchies = cv.findContours(
    thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(f'{len(contours)} countour(s) found')

# cv.destroyAllWindows()


def get_contour_areas(contours):

    all_areas = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        all_areas.append(area)

    return all_areas


sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)


largest_item = sorted_contours[0]

cv.drawContours(resized_image, largest_item, -1, (255, 0, 0), 10)
cv.waitKey(0)
cv.imshow('Largest Object', resized_image)


cv.waitKey(0)
# cv.destroyAllWindows()


#cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
#cv.imshow('Contours drawn', blank)

# Use canny first and then contours vs threshold (has disadvantages)
# cv.waitKey(0)
