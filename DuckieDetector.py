import cv2
import numpy as np

class DuckieDetector:

    def __init__(self):
        print("DuckieDetector Created!")

    def detectTest(self):
        image = cv2.imread("./dataset/frames/frame_002910.png")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #H :[10, 25], S: [100, 255], and V: [20, 255]

        lower_red = np.array([20,100,100])
        upper_red = np.array([30,255,255])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        cv2.imshow("Duckie Test Frame", image)
        cv2.imshow("Duckie Test Frame Mask", mask)

        cv2.waitKey(0)

    def checkContour(self, c):
    	# approximate the contour
        size = cv2.contourArea(c) # Calculate it's size
        peri = cv2.arcLength(c, True) # Calculate it's perimether
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        return size > 500 and len(approx) > 6

    def detect(self, frame):
        # Load Example Image
        #duckieImg = cv2.imread("", cv2.IMREAD_UNCHANGED)

        open_cv_image = np.array(frame)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        #cv2.imshow("Duckie Camera Frame", open_cv_image)

        #########

        hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)

        #H :[10, 25], S: [100, 255], and V: [20, 255]

        lower_red = np.array([15,100,100])
        upper_red = np.array([35,255,255])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        cv2.imshow("Duckie Frame Mask", mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contours

        validContours = []
        for c in contours:
            if self.checkContour(c):
                validContours.append(c)

        for c in validContours:
            size = cv2.contourArea(c)
            print("Area: " + str(size))

        cv2.drawContours(open_cv_image, validContours, -1,(0,255,0),3)

        cv2.imshow("Duckie Frame", open_cv_image)

        cv2.waitKey(1)

        #cv2.imwrite('example.png', open_cv_image)
