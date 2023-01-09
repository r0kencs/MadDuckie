import cv2
import numpy as np

def region(image):
    height, width = image.shape

    triangle = np.array([
                       [(0, height), (int(width * 1 / 4), int(height / 3)), (int(width * 3 / 4), int(height / 3)), (width, height)]
                       ])

    mask = np.zeros_like(image)

    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def average(image, lines):
    left = []
    right = []

    if lines is not None:
      for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
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
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

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

class LaneDetector:
    def detect(self, frame):
        open_cv_image = np.array(frame)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        img_hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([25, 100, 100], dtype='uint8')
        upper_yellow = np.array([25, 255, 255], dtype='uint8')

        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(gray, 150, 255)
        mask_ym = cv2.bitwise_or(mask_white, mask_yellow)
        mask_ym_image = cv2.bitwise_and(blur, mask_ym)

        region_image = region(mask_ym_image)

        edges = cv2.Canny(region_image, 50, 150)

        cv2.imshow("Edges", edges)

        lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)

        if lines is None:
            return

        for line in lines:
            x1, x2, y1, y2 = line[0]
            cv2.line(open_cv_image, (x1, x2), (y1, y2), (255, 0, 0), 3)

        print('Lines No: ' + str(len(lines)))

        averaged_lines = average(open_cv_image, lines)
        black_lines = display_lines(open_cv_image, averaged_lines)

        lanes = cv2.addWeighted(open_cv_image, 0.8, black_lines, 1, 1)

        cv2.imshow("Lane Detection", lanes)
        cv2.waitKey(1)
