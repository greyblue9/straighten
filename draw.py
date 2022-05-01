import cv2 as cv
import numpy as np
blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)
#img = cv.imread('Images/First.jpg')
#cv.imshow('First', img)

# 1. Paint the image a certain colour

#blank[200:300, 300:400] = 0, 0, 255
#cv.imshow('Green', blank)

# 2. Draw rectangle

Rectangle = cv.rectangle(
    blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=1)

cv.imshow('Rectangle', blank)

# 3. Draw circle

# cv.circle(blank, (250, 250), 48, (0, 0, 255), thickness=-1)
# cv.imshow('Circle', blank)

# # 4. Draw line

# cv.line(blank, (100, 250), (300, 400),
#         (250, 250, 250), thickness=3)
# cv.imshow('Line', blank)

# 5. Write text

cv.putText(blank, 'Hello', (225, 225), cv.FONT_HERSHEY_COMPLEX,
           1.0, (0, 255, 0), thickness=2)
cv.imshow('Text', blank)

cv.waitKey(0)
