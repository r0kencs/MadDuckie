import cv2
import numpy as np
import math

img = cv2.imread('screen.png')

def region(image):
    height, width = image.shape

    """triangle = np.array([
                       [(0, height), (int(width * 1 / 10), int(height / 2)), (int(width * 9 / 10), int(height / 2)), (width, height)]
                       ])"""

    triangle = np.array([
                       [(0, height), (0, int(height / 2)), (width, int(height / 2)), (width, height)]
                       ])

    mask = np.zeros_like(image)

    mask = cv2.fillPoly(mask, triangle, 255)

    #cv2.imshow("Mask", mask)

    mask = cv2.bitwise_and(image, mask)
    return mask

def drawHoughLines(image, lines):
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
            cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    return image

def drawHoughLinesP(image, lines):
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    return image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 150, 255, L2gradient = True)

cv2.imshow("Edges", edges)

region_image = region(edges)

cv2.imshow("Region", region_image)

lines = cv2.HoughLines(region_image, 1, np.pi / 180, 150, None, 0, 0)
linesP = cv2.HoughLinesP(region_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=1, maxLineGap=100)

imgCopy = img.copy()

img = drawHoughLines(img, lines)
imgCopy = drawHoughLinesP(imgCopy, linesP)

cv2.imshow("Hough Lane Detection", img)

cv2.imshow("Hough Probabilistic Lane Detection", imgCopy)

cv2.waitKey(0)
