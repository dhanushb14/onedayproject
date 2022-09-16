import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5','yolov5s' )
# img = 'https://ultralytics.com/images/zidane.jpg'
#
# results = model(img)
# print(results)
#
# %matplotlib inline
# plt.imshow(np.squeeze(results.render()))
# plt.show()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # make detection
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




