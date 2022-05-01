import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Create a multi plot
f, axarr = plt.subplots(2,3, sharex=True)
img = cv2.imread('Images/Second.jpg')
resized = cv2.resize(img, (250,250), interpolation = cv2.INTER_AREA)
# Show source image
axarr[0,0].imshow(resized)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
edges = cv2.Canny(blur_gray,50,150,apertureSize = 3)
# Show first edges image
axarr[0,1].imshow(edges)
h, w = edges.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
# Floodfill from point (0, 0)
cv2.floodFill(edges, mask, (0,0), 123);
# Show the flood fill image
axarr[0,2].imshow(edges)
floodfill = edges.copy()
bg = np.zeros_like(edges)
bg[edges == 123] = 255
# Show the masked image
axarr[1,0].imshow(bg)
bg = cv2.blur(bg, (3,3))
edges = cv2.Canny(bg,50,150,apertureSize = 3)
# Show the edges after masking
axarr[1,1].imshow(edges)

min_line_length = 50  
max_line_gap = 20

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

import math
lines = cv2.goodFeaturesToTrack(edges, 4, 0.7, 2)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(resized, pt1, pt2, (123,234,123), 2, cv2.LINE_AA)

xy = tuple(intersection(lines[0], lines[1])[0])
resized = cv2.circle(resized, xy, 5, 255, 2)
cv.imshow('output/edges.png', resized)
# Show the image with the corner
axarr[1,2].imshow(resized)
# Add titles
axarr[0,0].set_title('Source Image')
axarr[0,1].set_title('Edges')
axarr[0,2].set_title('Floodfill')
axarr[1,0].set_title('Masking')
axarr[1,1].set_title('Edges after masking')
axarr[1,2].set_title('Hough Lines')
# Clean up
axarr[0,0].axis('off')
axarr[0,1].axis('off')
axarr[1,0].axis('off')
axarr[1,1].axis('off')
axarr[1,2].axis('off')
axarr[0,2].axis('off')
plt.show()