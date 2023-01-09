import cv2
import numpy as np

img = cv2.imread('screen.png')

def region(image):
    height, width = image.shape

    triangle = np.array([
                       [(0, height - 100), (int(width * 1 / 3), int(height / 3)), (int(width * 2 / 3), int(height / 3)), (width, height - 100)]
                       ])

    """triangle = np.array([
                       [(0, height), (0, 0), (width, 0), (width, height)]
                       ])"""

    mask = np.zeros_like(image)

    mask = cv2.fillPoly(mask, triangle, 255)

    mask = cv2.bitwise_and(image, mask)

    cv2.imshow("Mask", mask)
    return mask

def drawLine(image, line):
    x1, x2, y1, y2 = line[0]
    result = cv2.line(image, (x1, x2), (y1, y2), (255, 0, 0), 2)
    return result

def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []

    y_bottom = 540
    y_upper = 315

    slope = 0
    b = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            # test and filter values to slope
            if ((y2-y1)/(x2-x1)) > 0.5 and ((y2-y1)/(x2-x1)) < 0.8:

                slope = ((y2-y1)/(x2-x1))
                b = y1 - slope*x1

                x_bottom_pos.append((y_bottom - b)/slope)
                x_upper_pos.append((y_upper - b)/slope)

            elif ((y2-y1)/(x2-x1)) < -0.5 and ((y2-y1)/(x2-x1)) > -0.8:

                slope = ((y2-y1)/(x2-x1))
                b = y1 - slope*x1

                x_bottom_neg.append((y_bottom - b)/slope)
                x_upper_neg.append((y_upper - b)/slope)

    # a new 2d array with means
    lines_mean = np.array([[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper))],
                            [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper))]])

    # Draw the lines
    for i in range(len(lines_mean)):
        cv2.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 50, 150)

cv2.imshow("Edges", edges)

region_image = region(edges)

cv2.imshow("Region", region_image)

lines = cv2.HoughLinesP(region_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=1000)

for line in lines:
    img = drawLine(img, line)

print('Lines No: ' + str(len(lines)))

draw_lines(img, lines, color=[0, 0, 255], thickness=2)

cv2.imshow("Lane Detection", img)
cv2.waitKey(0)
