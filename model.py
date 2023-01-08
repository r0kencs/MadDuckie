import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = 'https://www.youloveit.com/uploads/posts/2022-12/1670514221_youloveit_com_wednesday_addams_profile_images18.jpg'

results = model(img)
results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()
