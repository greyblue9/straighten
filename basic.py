import cv2 as cv
img = cv.imread('Images/First.jpg')
#cv.imshow('Image', img)


def rescaleFrame(frame, scale=0.125):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_img = rescaleFrame(img)
cv.imshow('Image', resized_img)


# 1. Converting to gray scale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# cv.imshow('Gray', gray)

# 2. Gaussian Blur
blur = cv.GaussianBlur(resized_img, (7, 7), cv.BORDER_DEFAULT)
#cv.imshow('Blur', blur)
#resized_image = rescaleFrame(blur)
#cv.imshow('Blur', resized_image)

# 3. Edge cascade - Canny Edge
#canny = cv.Canny(resized_img, 50, 50)
#cv.imshow('Canny Edges', canny)

# 4. Dialate Image
#dialted = cv.dilate(canny, (7, 7), iterations=3)
#cv.imshow('Dialated', dialted)
# 5. Eroding
#eroded = cv.erode(dialted, (7, 7), iterations=3)
#cv.imshow('Eroded', eroded)

# 6. Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('resized', resized)
canny = cv.Canny(resized, 50, 50)
cv.imshow('Canny Edges', canny)

# 7. Cropping
croped = img[50:200, 200:400]
cv.imshow('Cropped', croped)
cv.waitKey(0)
