from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

yolov7 = YOLOv7()
yolov7.load('C:/Users/freef/Desktop/theos/easy-yolov7-main/coco.weights', classes='C:/Users/freef/Desktop/theos/easy-yolov7-main/coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

webcam = cv2.VideoCapture(0)

if webcam.isOpened() == False:
	print('[!] error opening the webcam')

try:
    while webcam.isOpened():
        ret, frame = webcam.read()
        if ret == True:
            detections = yolov7.detect(frame, track=True)
            detected_frame = draw(frame, detections)
            print(json.dumps(detections, indent=4))
            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)
        else:
            break
except KeyboardInterrupt:
    pass

webcam.release()
print('[+] webcam closed')
yolov7.unload()