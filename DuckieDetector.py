import cv2
import numpy as np

class DuckieDetector:

    def __init__(self):
        print("DuckieDetector Created!")

    def detect(self, frame):
        # Load Example Image
        #duckieImg = cv2.imread("", cv2.IMREAD_UNCHANGED)

        open_cv_image = np.array(frame)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        cv2.imshow("Duckie Camera Frame", open_cv_image)

        cv2.waitKey(1)

        #cv2.imwrite('example.png', open_cv_image)
