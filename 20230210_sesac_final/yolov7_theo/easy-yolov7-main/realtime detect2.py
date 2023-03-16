from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
import numpy as np
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

yolov7 = YOLOv7()
yolov7.load('C:/Users/freef/Desktop/theos/easy-yolov7-main/coco.weights', classes='C:/Users/freef/Desktop/theos/easy-yolov7-main/coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

webcam = cv2.VideoCapture(0)
if webcam.isOpened() == False:
    print('[!] error opening the webcam')

lines = {}
arrow_lines = []
arrow_line_length = 50

width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(webcam.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# initialize variables for output video
current_obj_id = None
output = None

try:
    with tqdm() as pbar:
        while webcam.isOpened():
            ret, frame = webcam.read()
            if ret == True:
                detections = yolov7.detect(frame, track=True)
                detected_frame = frame

                for detection in detections:
                    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

                    if 'id' in detection:
                        detection_id = detection['id']
                        
                        if current_obj_id != detection_id:
                            # if a new object is detected, create a new output video file
                            if output is not None:
                                output.release()
                            output_path = f"C:/Users/freef/Desktop/theos/easy-yolov7-main/output{detection_id}.mp4"
                            width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = int(webcam.get(cv2.CAP_PROP_FPS))
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            current_obj_id = detection_id

                            # initialize lines and arrow_lines for the new object
                            lines[detection_id] = {'points':[], 'arrows':[], 'color':color}
                            arrow_lines = []
                        else:
                            detection['color'] = lines[detection_id]['color']

                        lines[detection_id]['points'].append(np.array([detection['x'] + detection['width']/2, detection['y'] + detection['height']/2], np.int32))
                        points = lines[detection_id]['points']

                        if len(points) >= 2:
                            arrow_lines = lines[detection_id]['arrows']
                            if len(arrow_lines) > 0:
                                distance = np.linalg.norm(points[-1] - arrow_lines[-1]['end'])
                                if distance >= arrow_line_length:
                                    start = np.rint(arrow_lines[-1]['end'] - ((arrow_lines[-1]['end'] - points[-1])/distance)*10).astype(int)
                                    arrow_lines.append({'start':start, 'end':points[-1]})
                            else:
                                distance = 0
                                arrow_lines.append({'start':points[-2], 'end':points[-1]})

                for line in lines.values():
                    arrow_lines = line['arrows']
                    for arrow_line in arrow_lines:
                        frame = cv2.arrowedLine(frame, arrow_line['start'], arrow_line['end'], line['color'], 2, line_type=cv2.LINE_AA)

                frame = draw(frame, detections)
                
                if output is not None:
                    # write the frame to the output video file
                    output.write(frame)
                
                cv2.imshow('frame',frame) # display the frame on the screen
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                pbar.update(1)
            else:
                break
except Exception as e:
    print("An exception occurred:", e)
finally:
    webcam.release()
    if output is not None:
        output.release()
    cv2.destroyAllWindows()