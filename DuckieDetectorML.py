import cv2
import numpy as np
import torch
import math

MODEL_PATH = 'model/exp3/weights/best.pt'

class DuckieDetectorML:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

    def detect(self, frame):
        open_cv_image = np.array(frame)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

        results = self.model(img)

        results.render()
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('YOLOv5', out)

        happy_results = results.xyxy[0].cpu().numpy()

        distances = []

        for result in happy_results:
            x1, y1, x2, y2, confidence, c_name = result
            #print(f"x1: {x1} y1: {y1} x2: {x2} y2: {y2} confidence: {confidence} class: {c_name}")
            d1 = math.dist([320, 480], [x1, y1])
            d2 = math.dist([320, 480], [x2, y2])
            distance = min([d1, d2])
            distances.append(distance)
            #print(f"Distance: {distance}")

        cv2.waitKey(1)

        if len(distances) > 0:
            min_d = min(distances)
        else:
            min_d = None

        return min_d
