import cv2
import numpy as np
import torch

MODEL_PATH = 'yolov5/runs/train/exp/weights/last.pt'

class DuckieDetectorML:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

    def detect(self, frame):
        open_cv_image = np.array(frame)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

        results = model(img)

        results.render()
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('YOLOv5', out)

        cv2.waitKey(1)
