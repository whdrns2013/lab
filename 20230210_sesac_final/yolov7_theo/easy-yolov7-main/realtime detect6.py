from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
from datetime import datetime
import numpy as np
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

yolov7 = YOLOv7()
yolov7.load('C:/Users/freef/Desktop/theos/easy-yolov7-main/coco.weights', classes='C:/Users/freef/Desktop/theos/easy-yolov7-main/coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print('[!] error opening the webcam')

# initialize variables for output video and log file
lines = {}
outputs = {}
log_files = {}
current_obj_ids = []
arrow_lines = {}
log_data = {}
arrow_line_length = 50

width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(webcam.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

try:
    with tqdm() as pbar:
        while webcam.isOpened():
            ret, frame = webcam.read()
            if ret:
                detections = yolov7.detect(frame, track=True)
                detected_frame = frame

                for detection in detections:
                    if 'id' in detection:
                        detection_id = detection['id']

                        # create new output file and log file if a new object is detected
                        if detection_id not in current_obj_ids:
                            # create new output and log file for the new object
                            now = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")
                            output_path = f"C:/Users/freef/Desktop/theos/easy-yolov7-main/output{detection_id}_{now}.mp4"
                            log_path = f"C:/Users/freef/Desktop/theos/easy-yolov7-main/log/{detection_id}_{now}_log.txt"
                            width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = int(webcam.get(cv2.CAP_PROP_FPS))
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            current_obj_ids.append(detection_id)
                            outputs[detection_id] = output

                            # initialize lines and arrow_lines for the new object
                            lines[detection_id] = {'points':[], 'arrows':[], 'color':(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))}
                            arrow_lines[detection_id] = []
                            log_data[detection_id] = []

                            # create log file
                            log_file = open(log_path, 'w')
                            log_files[detection_id] = log_file

                        # draw arrow lines for the object
                        lines[detection_id]['points'].append(np.array([detection['x'] + detection['width']/2, detection['y'] + detection['height']/2], np.int32))
                        points = lines[detection_id]['points']

                        if len(points) >= 2:
                            arrow_lines_list = arrow_lines[detection_id]
                            if len(arrow_lines_list) > 0:
                                distance = np.linalg.norm(points[-1] - arrow_lines_list[-1]['end'])
                                if distance >= arrow_line_length:
                                    start = np.rint(arrow_lines_list[-1]['end'] - ((arrow_lines_list[-1]['end'] - points[-1])/distance)*10).astype(int)
                                    arrow_lines_list.append({'start':start, 'end':points[-1]})
                            else:
                                distance = 0
                                arrow_lines_list.append({'start':points[-2], 'end':points[-1]})

                            # draw arrow lines for the object on the frame
                            for arrow_line in arrow_lines_list:
                                frame = cv2.arrowedLine(frame, arrow_line['start'], arrow_line['end'], lines[detection_id]['color'], 2, line_type=cv2.LINE_AA)

                        # write detection data to log file
                        log_data[detection_id].append(f"x={detection['x']}, y={detection['y']}, w={detection['width']}, h={detection['height']}, t={webcam.get(cv2.CAP_PROP_POS_MSEC)}\n")

                        # set color for the detection box and draw it on the frame
                        detection['color'] = lines[detection_id]['color']
                        detected_frame = draw(detected_frame, [detection])

                # write the frame with detection boxes to the output video file for the object
                output = outputs.get(detection_id)
                if output is not None:
                    output.write(detected_frame)

                # write log data to log file for the object
                log_file = log_files.get(detection_id)
                if log_file is not None:
                    log_file.writelines(log_data[detection_id])

                # display the frame on the screen
                cv2.imshow('frame', frame)

                # break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pbar.update(1)
            else:
                break
except Exception as e:
    print("An exception occurred:", e)
finally:
    # release the resources
    webcam.release()

    # release the output video file and log file for each object
    for output in outputs.values():
        if output is not None:
            output.release()

    for log_file in log_files.values():
        if log_file is not None:
            log_file.close()

    cv2.destroyAllWindows()