# Read input
import cv2 as cv
import numpy as np
import itertools
color = cv.imread('Images/Third.jpg', cv.IMREAD_COLOR)


def rescaleFrame(frame, scale=0.125):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_image = rescaleFrame(color)
cv.imshow('Image', resized_image)
gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow('output/gray.png', gray)
# cv2.imwrite('output/thresh.png', thresh)
# Edge detection
edges = cv.Canny(gray, 100, 200)
# Save the edge detected image


"""
Script contains functions to process the lines in polar coordinate system
returned by the HoughLines function in OpenCV
Line equation from polar to cartesian coordinates
x = rho * cos(theta)
y = rho * sin(theta)
x, y are at a distance of rho from 0,0 at an angle of theta
Therefore,
    m = (y - 0) / (x - 0)
    Using the values of x, y, and m
    b = y - m * x
Python 3.6 was used to compile and test the code.
"""


def polar2cartesian(rho: float, theta_rad: float, rotate90: bool = False):
    """
    Converts line equation from polar to cartesian coordinates
    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line
    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x

    return m, b


def solve4x(y: float, m: float, b: float):
    """
    From y = m * x + b
         x = (y - b) / m
    """
    if np.isclose(m, 0.0):
        return 0.0
    if m is np.nan:
        return b
    return (y - b) / m


def solve4y(x: float, m: float, b: float):
    """
    y = m * x + b
    """
    if m is np.nan:
        return b
    return m * x + b


def intersection(m1: float, b1: float, m2: float, b2: float):
    # Consider y to be equal and solve for x
    # Solve:
    #   m1 * x + b1 = m2 * x + b2
    x = (b2 - b1) / (m1 - m2)
    # Use the value of x to calculate y
    y = m1 * x + b1

    return int(round(x)), int(round(y))


def line_end_points_on_image(rho: float, theta: float, image_shape: tuple):
    """
    Returns end points of the line on the end of the image
    Args:
        rho: input line rho
        theta: input line theta
        image_shape: shape of the image
    Returns:
        list: [(x1, y1), (x2, y2)]
    """
    m, b = polar2cartesian(rho, theta, True)

    end_pts = []

    if not np.isclose(m, 0.0):
        x = int(0)
        y = int(solve4y(x, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))
            x = int(image_shape[1] - 1)
            y = int(solve4y(x, m, b))
            if point_on_image(x, y, image_shape):
                end_pts.append((x, y))

    if m is not np.nan:
        y = int(0)
        x = int(solve4x(y, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))
            y = int(image_shape[0] - 1)
            x = int(solve4x(y, m, b))
            if point_on_image(x, y, image_shape):
                end_pts.append((x, y))

    return end_pts


def hough_lines_end_points(lines: np.array, image_shape: tuple):
    """
    Returns end points of the lines on the edge of the image
    """
    if len(lines.shape) == 3 and \
            lines.shape[1] == 1 and lines.shape[2] == 2:
        lines = np.squeeze(lines)
    end_pts = []
    for line in lines:
        rho, theta = line
        end_pts.append(
            line_end_points_on_image(rho, theta, image_shape))
    return end_pts


def hough_lines_intersection(lines: np.array, image_shape: tuple):
    """
    Returns the intersection points that lie on the image
    for all combinations of the lines
    """
    if len(lines.shape) == 3 and \
            lines.shape[1] == 1 and lines.shape[2] == 2:
        lines = np.squeeze(lines)
    lines_count = len(lines)
    intersect_pts = []
    for i in range(lines_count - 1):
        for j in range(i + 1, lines_count):
            m1, b1 = polar2cartesian(lines[i][0], lines[i][1], True)
            m2, b2 = polar2cartesian(lines[j][0], lines[j][1], True)
            x, y = intersection(m1, b1, m2, b2)
            if point_on_image(x, y, image_shape):
                intersect_pts.append([x, y])
    return np.array(intersect_pts, dtype=int)


def point_on_image(x: int, y: int, image_shape: tuple):
    """
    Returns true is x and y are on the image
    """
    return 0 <= y < image_shape[0] and 0 <= x < image_shape[1]


def cyclic_intersection_pts(pts):
    """
    Sorts 4 points in clockwise direction with the first point been closest to 0,0
    Assumption:
        There are exactly 4 points in the input and
        from a rectangle which is not very distorted
    """
    pts = np.array(pts)
    #if pts.shape[0] != 4:
    #    return None

    # Calculate the center
    center = np.mean(pts, axis=0)

    # Sort the points in clockwise
    cyclic_pts = [
        # Top-left
        pts[np.where(np.logical_and(pts[:, 0] < center[0],
                     pts[:, 0:1] < center[0:1]))[0][0], :],
        # Top-right
        pts[np.where(np.logical_and(pts[:, 0] > center[0],
                     pts[:, 0:1] < center[0:1]))[0][0], :],
        # Bottom-Right
        pts[np.where(np.logical_and(pts[:, 0] > center[0],
                     pts[:, 0:1] > center[0:1]))[0][0], :],
        # Bottom-Left
        pts[np.where(np.logical_and(pts[:, 0] < center[0],
                     pts[:, 0:1] > center[0:1]))[0][0], :]
    ]

    return np.array(cyclic_pts)

def drawHoughLines(image, lines, output):
    out = image.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imwrite(output, out)


def hough_lines_intersection(self):
    """Finds the intersections between groups of lines."""
    lines = self
    intersections = []
    group_lines = itertools.combinations(range(len(lines)), 2)
    x_in_range = lambda x: 0 <= x <= resized_image.shape[1]
    y_in_range = lambda y: 0 <= y <= resized_image.shape[0]

    for i, j in group_lines:
      line_i, line_j = lines[i][0], lines[j][0]

      if 80.0 < _get_angle_between_lines(None, line_i, line_j) < 100.0:
          int_point = _intersection(None, line_i, line_j)

          if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]): 
              intersections.append(int_point)
    output_process = False
    if output_process: _draw_intersections(intersections)

    return intersections
from math import atan, degrees, pi      
def _get_angle_between_lines(self, line_1, line_2):
    rho1, theta1 = line_1
    rho2, theta2 = line_2
    # x * cos(theta) + y * sin(theta) = rho
    # y * sin(theta) = x * (- cos(theta)) + rho
    # y = x * (-cos(theta) / sin(theta)) + rho
    m1 = -(np.cos(theta1) / np.sin(theta1))
    m2 = -(np.cos(theta2) / np.sin(theta2))
    return abs(atan(abs(m2-m1) / (1 + m2 * m1))) * (180 / np.pi)

    
def _intersection(self, line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
      [np.cos(theta1), np.sin(theta1)],
      [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

margin = len(edges) // 16
edges[0:margin, :] = [[0] * edges.shape[1]] * margin
edges[-margin:, :] = [[0] * edges.shape[1]] * margin
edges[:, 0:margin] = [[0] * margin] * edges.shape[0]
edges[:, -margin:] = [[0] * margin] * edges.shape[0]
cv.imshow('output/edges.png', edges)


# Detect lines using hough transform
polar_lines = cv.HoughLines(edges, 1, np.pi / 180, 150)
drawHoughLines(color, polar_lines, 'output/houghlines.png')
# Detect the intersection points
intersect_pts = hough_lines_intersection(polar_lines)
# Sort the points in cyclic order
intersect_pts = cyclic_intersection_pts(intersect_pts)
# Draw intersection points and save
out = color.copy()
for pts in intersect_pts:
    cv.rectangle(out, (pts[0][0] - 1, pts[0][1] - 1),
                 (pts[0][0] + 1, pts[0][1] + 1), (0, 0, 255), 2)
cv.imwrite('output/intersect_points.png', out)
cv.waitKey(0)
cv.imshow('intersect_points')
