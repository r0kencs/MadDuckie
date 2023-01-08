import sys
import json
import numpy as np
import cv2

annotationFile = './dataset/annotation/final_anns.json'
framePath = './dataset/frames/'

target = './dataset/annotation/values/'

with open (annotationFile) as f:
    data = json.load(f)

for name in data.keys():
    imageName = framePath + name
    height, width = cv2.imread(imageName).shape[:2]
    #print('Name: ' + imageName)
    #print('Size: ' + str(width) + 'x' + str(height))

    result = []

    i = 0
    for i, annotation in enumerate(data[name]):
        bbox = [annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2], annotation['bbox'][1] + annotation['bbox'][3]]

        annotationResult = {
            "id": "result" + str(i),
            "type": "rectanglelabels",
            "from_name": "label", "to_name": "image",
            "original_width": width, "original_height": height,
            "image_rotation": 0,
            "value": {
              "rotation": 0,
              "x": bbox[0] / width * 100.0, "y": bbox[1] / height * 100.0,
              "width": (bbox[2] - bbox[0]) / width * 100.0, "height": (bbox[3] - bbox[1]) / height * 100.0,
              "rectanglelabels": [annotation['cat_name'].capitalize()]
            }
        }

        result.append(annotationResult)

    data = {
        "image": name
    }

    predictions = [{
        "result": result
    }]

    task = [{
        "data": data,
        "annotations": predictions
    }]

    #print(json.dumps(task))

    with open(target + 'json_data.json', 'w') as outfile:
        json.dump(task, outfile)

    break
