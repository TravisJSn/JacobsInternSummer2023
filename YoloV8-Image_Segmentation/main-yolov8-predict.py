from ultralytics import YOLO
import cv2
import torch
import numpy as np

model = YOLO('/weights/best.pt')
results = model('', imgsz=640)
img = cv2.imread('.jpg')

for result in results:
    for mask in result.masks:
        m = torch.squeeze(mask.data)
        composite = torch.stack((m, m, m), 2)
        tmp = img * composite.cpu().numpy().astype(np.uint8)
        cv2.imshow("result", composite)
        cv2.waitKey(0)