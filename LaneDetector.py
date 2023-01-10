import cv2
import numpy as np
import math

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

def getCorrectLines(lines):
    if lines is None:
        return None

    correctLines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)

        if slope > 0:
            correctLines.append(line)
        #print(f"x1: {x1} y1: {y1} x2: {x2} y2: {y2} Slope: {slope}")

    return correctLines

def average(image, lines):
    left = []
    right = []

    if lines is not None:
      for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)

    finalLines = []

    #create lines based on averages calculates
    if left_avg is not None and not np.isnan(left_avg).all() :
        left_line = make_points(image, left_avg)
        finalLines.append((left_line, False))

    if right_avg is not None and not np.isnan(right_avg).all():
        right_line = make_points(image, right_avg)
        finalLines.append((right_line, True))

    return finalLines

def make_points(image, average):
    print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            (x1, y1, x2, y2), right = line
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def unpackLines(lines):
    res = [None, None]

    if lines is not None:
        for line in lines:
            (x1, y1, x2, y2), right = line
            if right:
                res[1] = np.array([x1, y1, x2, y2])
            else:
                res[0] = np.array([x1, y1, x2, y2])

    return res


class LaneDetector:
    def detect(self, frame):
        open_cv_image = np.array(frame)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        img = open_cv_image.copy()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_white = np.array([20,60,100],np.uint8)
        upper_white = np.array([80,255,255],np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        res = cv2.bitwise_and(img, img, mask=white_mask)

        res2 = cv2.bitwise_xor(img, res)

        cv2.imshow('res', res)
        cv2.imshow('res2',res2)

        gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

        cv2.imshow("thresh", thresh)

        blur = cv2.GaussianBlur(thresh, (5, 5), 0)

        edges = cv2.Canny(blur, 0, 255, L2gradient = True)

        region_image = region(edges)

        canny1 = cv2.bitwise_or(gray, edges)

        #lines = cv2.HoughLines(region_image, 1, np.pi / 180, 150, None, 0, 0)
        linesP = cv2.HoughLinesP(region_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=1, maxLineGap=100)

        #linesP = getCorrectLines(linesP)

        imgCopy = img.copy()

        averaged_lines = average(imgCopy, linesP)

        #img = drawHoughLines(img, lines)
        #imgCopy = drawHoughLinesP(imgCopy, averaged_lines)

        black_lines = display_lines(imgCopy, averaged_lines)

        #cv2.imshow("Hough Lane Detection", img)

        lanes = cv2.addWeighted(imgCopy, 0.5, black_lines, 1, 1)
        cv2.imshow("Hough Probabilistic Lane Detection", lanes)

        res = unpackLines(averaged_lines)

        cv2.waitKey(1)

        return res
