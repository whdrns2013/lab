'''



% 코드 버전 내역
ver1.0.0 : 객체 탐지 및 경로 추적 기능
ver1.1.0 : danger_range 내 탐지된 경우 알람 재생
ver1.1.1 : danger range 를 코드 실행 시 사용자가 설정 가능

% ver1.1.2 주요 사항
- 메서드 리팩토링
- 로그 저장값 변경 (timestamp 추가)
- 로그 데이터프레임화 및 csv 저장 구현

% To Do
- 객체 경로 추적 메서드 뜯어보기
- 객체 탐지시 로그 저장 기능 테스트
- 객체 탐지시 이미지 저장 기능 테스트
- gif 만들기 메서드 추가
- log의 dataframe 화
- AWS RDS 파일 전송 구현
- AWS S3 파일 전송 구현


'''


import argparse # 입력 인수를 구문 분석
import time # 시간 관련 함수를 사용하기 위한 라이브러리
from pathlib import Path # 파일 경로와 관련된 함수를 사용하기 위한 라이브러리
import cv2 # OpenCV 라이브러리를 사용하기 위한 라이브러리
import torch # PyTorch 라이브러리를 사용하기 위한 라이브러리
import torch.backends.cudnn as cudnn # PyTorch에서 CUDA 연산을 가속화하기 위한 라이브러리
from numpy import random # 난수 생성과 관련된 함수를 사용하기 위한 라이브러리
from datetime import datetime # 날짜와 시간과 관련된 함수를 사용하기 위한 라이브러리

from models.experimental import attempt_load # 모델을 로드하는 함수가 들어있다.
from utils.datasets import LoadStreams, LoadImages # 이미지 또는 비디오 스트림을 로드한다.
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path # non_max_suppression : NMS를 적용하여 객체 탐지를 정제한다.
                # 다양한 유틸리티 함수들이 들어 있다. 이미지 크기를 체크하거나, NMS를 적용하는 등의 작업을 수행한다.
from utils.plots import plot_one_box # 객체 탐지 결과를 
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import playsound
import threading
from sort import * # 객체를 추적한다.

import os
import pandas as pd


# 종혁 추가 : global 변수
alarm_path = '/Users/jongya/Desktop/Workspace/lab/20230210_sesac_final/yolo7-codingbug/yolov7/beep.wav'



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def saving_images(img_path, timestamp, im0):
    if os.path.exists(img_path):
        pass
    else:
        os.makedirs(img_path)
    
    cv2.imwrite(img_path + '/' + timestamp + '.jpg', im0)
    
# 종혁 : 테스트
def sounding(soundpath):
    playsound.playsound(soundpath)

def run_sounding(soundpath):
    t = threading.Thread(target = sounding, args = (soundpath, ))
    t.start()
    
def saving_videos(vid_writer, vid_path, save_dir, im0, vid_cap=None):
    if vid_path is None:
        now = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분 %S초")
        save_path = str(save_dir / f"{now}.mp4")
        vid_path = save_path
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
        if vid_cap:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            fps, w, h = 30, im0.shape[1], im0.shape[0]
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    vid_writer.write(im0)

    return vid_writer, vid_path

def saving_logs(txt_path, event_name, s, timestamp):
    with open(txt_path + '/' + event_name + '.txt', 'w') as f:
        f.write(timestamp + s + '\n')
        
def make_log_dataframe(path, event_name):
    log_list = os.listdir(path)
    log_list = [path + '/' + log for log in log_list if log.endswith('.txt')]
    log_list.sort()
    result = []
    
    origin_col_name = ['event_name', 'detect_date', 'detect_time',
                       'object1_num', 'object1_name',
                       'object2_num', 'object2_name',
                       'object3_num', 'object3_name',]
    
    final_col_name = ['event_name', 'detect_date', 'detect_time',
                         'object1_name', 'object1_num',
                         'object2_name', 'object2_num',
                         'object3_name', 'object3_num',]
    
    for log in log_list:
        f = open(log, 'r')
        lines = f.readlines()
        
        for line in lines:
            line = line.replace('\n', '').replace(',', '').replace(':', '').split(' ')
            line.insert(0, event_name)
            del line[len(line) - 1]
            while len(line) < len(origin_col_name):
                line.append('')
            
            print(line)
            result.append(line)
    log_dataframe = pd.DataFrame(result, columns = origin_col_name)
    log_dataframe = log_dataframe[final_col_name]
    log_dataframe.to_csv(path + f'/logs_{event_name}.csv')
    print("log를 데이터프레임으로 변환 후 저장하였습니다.")
    
    

def draw_minimap(im0, bbox_xyxy, identities, categories, names, colors):
    height, width, _ = im0.shape
    minimap_size = (int(width * 0.2), int(height * 0.2))
    minimap = np.zeros((minimap_size[1], minimap_size[0], 3), dtype=np.uint8)

    if len(bbox_xyxy) > 0:
        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            color = colors[int(categories[i])]

            minimap_center_x = int(center_x * minimap_size[0] / width)
            minimap_center_y = int(center_y * minimap_size[1] / height)
            cv2.circle(minimap, (minimap_center_x, minimap_center_y), 2, color, -1)

    im0[height - minimap_size[1]:, width - minimap_size[0]:] = minimap

    return im0

"""Function to Draw Bounding boxes(객체 탐지 결과를 시각화 한다.)""" 
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img

def make_danger_range_display(danger_range_ratio, im0):
    danger_range = int(im0.shape[0] * danger_range_ratio)
    danger_range_img = np.zeros((im0.shape[0], im0.shape[1], 3), dtype=np.uint8)
    for i in range(11):
        recog = int(im0.shape[0]/10)
        cv2.line(im0, (0, recog*i), (int(im0.shape[1]/20), recog*i), (255, 0, 0), 5)
        cv2.line(im0, (int(im0.shape[1]), recog*i), (int(im0.shape[1] - im0.shape[1]/20), recog*i), (255, 0, 0), 5)
    cv2.line(im0, (int(im0.shape[1]/2), 0), (int(im0.shape[1]/2), im0.shape[0]), (255, 255, 255), 3)
    cv2.line(im0, (0, int(im0.shape[0]/2)), (im0.shape[1], int(im0.shape[0]/2)), (255, 255, 255), 3)
    cv2.rectangle(danger_range_img, (0, danger_range), (im0.shape[1], im0.shape[0]), (0, 0, 255), -1)
    im0 = cv2.addWeighted(im0, 1.0, danger_range_img, 0.2, 0.0)
    
    return im0


def detect(save_img=True):
    source = opt.source
    weights = opt.weights
    view_img = opt.view_img
    save_txt = opt.save_txt
    imgsz = opt.img_size
    trace = not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    danger_range_ratio = opt.danger_range_ratio # 종혁 : 추가 - danger range ratio 받아오기

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = 0
    ###################################
    
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print(det)
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels') + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path = str(save_dir / 'logs')
            img_path = str(save_dir / 'imgs')
            
            if os.path.exists(txt_path):
                pass
            else:
                os.makedirs(txt_path)
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    now = datetime.now()
                    event_name = now.strftime("%Y-%m-%d-%H-%M-%S.%f")
                    timestamp = now.strftime("%Y-%m-%d %H-%M-%S.%f")
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    saving_logs(txt_path, event_name, s, timestamp)
                    make_log_dataframe(txt_path, event_name)

                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                    
                    # 종혁 추가 : 경보
                    object_bot_center = [int((x1 + x2)/2), int(y2)]
                    danger_range_ratio = opt.danger_range_ratio
                    height = im0.shape[0]
                    danger_range = int(height * danger_range_ratio)
                    # x1 : 좌상단 x 좌표 / y1 : 좌상단 y 좌표 / x2 : 우하단 x 좌표 / y2 : 우하단 y 좌표
                    if object_bot_center[1] >= danger_range:
                        global alarm_path
                        run_sounding(alarm_path)
                
                if opt.track:
  
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    tracks =sort_tracker.getTrackers()
                    sort_tracker.color_list

                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if opt.show_track:
                            #loop over tracks
                            for t, track in enumerate(tracks):
                  
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])), 
                                                (int(track.centroidarr[i+1][0]),
                                int(track.centroidarr[i+1][1])),
                                track_color, thickness=opt.thickness) 
                                for i,_ in  enumerate(track.centroidarr) 
                                if i < len(track.centroidarr)-1 ] 
                else:
                    bbox_xyxy = dets_to_sort[:,:4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                
                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                im0 = draw_minimap(im0, bbox_xyxy, identities, categories, names, colors)
                
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d-%H-%M-%S.%f")
                
                saving_images(img_path, timestamp, im0)
                

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            ######################################################
            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

            #######################################################
            # if view_img:
            #     try:
            #         # 종혁 추가 : 보여주는 이미지에 danger range 추가
            #         danger_range = int(im0[0] * danger_ragne_ratio)
            #         danger_range_img = np.zeros((im0.shape[0], im0.shape[1], 3), dtype=np.uint8)
            #         cv2.rectangle(danger_range_img, (0, danger_range), (im0.shape[1], im0.shape[0]), (0, 0, 255), -1)
            #         cv2.addWeighted(im0, 1.0, danger_range_img, 0.4, 50.0)
            #         cv2.imshow(str(p), im0)
            #         cv2.waitKey(1)  # 1 millisecond
            #         print('이건 문제 없음')
            #     except:
            #         cv2.imshow(str(p), im0)
            #         cv2.waitKey(1)  # 1 millisecond
            #         print("그러면 이건?")

            if view_img:
                # 종혁 추가 : 화면에 danger_range 가 나오도록
                danger_range_ratio = 1.0 - opt.danger_range_ratio
                im0 = make_danger_range_display(danger_range_ratio, im0)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                print('이건 문제 없음')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if len(det) == 0:  # 탐지된 객체가 없으면
                        if vid_path is not None:  # 이전에 저장된 비디오 파일이 있으면 저장
                            vid_writer.release()
                            print(f"The video file {vid_path} is saved.")
                            vid_path = None

                    else:  # 새로운 비디오 파일 생성
                        vid_writer, vid_path = saving_videos(vid_writer, vid_path, save_dir, im0, vid_cap)

            # 종료 조건 추가
            if len(det) == 0 and dataset.mode != 'image':
                if webcam:
                    continue
                else:
                    if vid_path is not None:
                        vid_writer.release()
                        print(f"The video file {vid_path} is saved.")
                    cv2.waitKey(0)
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--save-image', action='store_true', help='save_image_and_log')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')
    parser.add_argument('--danger-range-ratio', type=float, default=0.5, help='danger range ratio, [0.0 -1.0]') # 종혁 추가 : danger range ratio 받아오기

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                       min_hits=2,
                       iou_threshold=0.2) 

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()