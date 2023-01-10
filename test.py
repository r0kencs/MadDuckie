import cv2
import numpy as np
import torch
import math

MODEL_PATH = 'model/exp/weights/last.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

open_cv_image = cv2.imread('lane.png')
img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

results = model(img)

#print(results[0])

#results.show()

results.render()
out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow('YOLOv5', out)

print(results.xyxy)

happy_results = results.xyxy[0].numpy()

#print(results.xyxy[0][0][0])

for result in happy_results:
    x1, y1, x2, y2, confidence, c_name = result
    print(f"x1: {x1} y1: {y1} x2: {x2} y2: {y2} confidence: {confidence} class: {c_name}")
    d1 = math.dist([320, 480], [x1, y1])
    d2 = math.dist([320, 480], [x2, y2])
    distance = min([d1, d2])
    print(f"Distance: {distance}")

cv2.waitKey(0)
