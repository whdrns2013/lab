'''
############################################################################
##########################  detect.py 명세서  ################################
############################################################################

1-1. 주요 기능
- YOLOv7 모델을 통한 객체 인식(탐지) 기능
- 객체 tracking (sort)
- 객체 인식(탐지) 결과를 이미지와 로그데이터로 자동 저장

1-2. 부가 기능
- 저장된 로그데이터를 RDS에 자동 업로드
- 저장된 이미지파일을 S3에 자동 업로드
- 이미지들을 자동으로 gif로 만드는 기능 (구현했으나, 실행되지 않게 막음)
- 객체가 위험구역 내에서 탐지되면 자동 경보 울림

% 명령어
--remove-old-track : 사라진 객체의 tracking 선을 제거합니다.
--danger-range 0.x : 객체가 화면의 x% 범위 내로 접근하면 알람이 울리도록 범위 설정
--conf-thres 0.y : 객체 인식 신뢰도
python detect_or_trackmbasic_1_3_3.py --weights wild5_retry.pt --no-trace --view-img --source 0 --track --show-track --unique-track-color --danger-range 0.5 --conf-thres 0.7 --detecting-time 10

% 코드 버전 내역
ver1.0.0 : 객체 탐지 및 경로 추적 기능
ver1.1.0 : danger_range 내 탐지된 경우 알람 재생
ver1.1.1 : danger range 를 코드 실행 시 사용자가 설정 가능
ver1.1.2 : 로그 데이터프레임화 및 csv 저장
ver1.1.3 : AWS RDS, S3 파일 전송 메서드 구현, gif 만들기 메서드 추가
ver1.1.4 : detect 메서드 내 minimap 메서드 삭제
ver1.2.0 : 객체가 사라진 후 tracking 선을 유지할지 말지 결정할 수 있게 업데이트 (sort.py 변경)
ver1.2.1 : 객체 경로 추적 선 화살표로 변경, 경로 추적점을 중앙 하단으로 변경
ver1.2.2 : tracking 메서드화
ver1.3.0 : 이미지 저장, 로그 저장, RDS, S3 저장 알고리즘 구현
ver1.3.1 : 이미지 저장, 로그 저장 알고리즘 개선
ver1.3.2 : trace와 실제 객체 위치 다른 오류 해결, 저장 및 알람 알고리즘 개선, 이벤트 종료시 trace 기록 삭제

% ver1.3.3 주요 사항
- 속도 개선 테스트

% 속도 개선 테스트
현황
- yolov7, detect1.3.2, best.pt 기준 : 최소 400ms ~ 최대 1700ms
- Inference : 400ms ~ 1700ms
- NMS(normalization) : 0.1 ~ 0.3ms
- operation(sort, 저장, 알람) : 0.0ms
현황 진단
- sort, 알람, 저장 등의 추가한 기능들은 속도에 영향이 거의 없음
- 즉, yolov7 자체의 추론 시간이 너무 길다는 게 문제.
속도 개선 테스트 1
- weight 교체 : 원래 best.pt(207MB) -> wild5_retry.pt(75MB)
- 유의미한 결과 : best.pt에서는 탐지 객체가 많아지면 속도가 1700ms(1.7초)까지 느려졌으나, wild5_retry에서는 최소 400ms ~ 최대 580ms 로 안정적
- 하지만, 사람을 야생동물로 잡는 문제는 여전히 발생 - 이 부분은 개선될 것으로 기대
결론
(1) weight 파일의 용량이 작을수록 처리 속도가 안정적
(2) 추가 기능(operation)은 처리 속도에 거의 영향을 끼치지 않음
(3) 2번에 따라, yolov5에 sort등 operation을 추가하는 게 성능 개선으로는 가장 합리적인 선택
(4) 하지만 이미 구축한 yolov7의 알고리즘을 yolov5에 최적화하는 데에는 1~2일의 시간이 필요할 것으로 예상되며, sort를 적용할 수 있을지도 미지수
(5) yolov7을 사용하려면 Inference, 즉 순정 yolov7 성능 튜닝이 필요하며
(6) 5번에 이어서, yolov7 자체의 성능 튜닝 없이는 double buffering을 적용한다 해도 성능 자체 개선은 되지 않으므로, 객체 추적은 여전히 400ms ~ 500ms interval로 진행됨 (0.4초 ~ 0.5초)

% To Do
- yolov7 처리 속도 개선
- 비디오 기록되도록
- event_type 1일 때, 객체가 danger-range 이내로 들어와도 알람이 울리지 않는 문제 개선
- 캠 ID 넣기
- 탐지 화면에서 객체 이름 제대로 나오도록
- 명세서 작성

% 우려점
- sort에 대해서 설명할 수가 없음
- sort 는 각 객체별 위치정보와 표시 색상을 정한 후, 이를 for문으로 돌면서 그려주는 것인데..
- yolo7 처리 속도 느림 / sort가 추가되었다고 해도, yolov5 interval 110~130ms, yolov7 interval 400~1700ms, 최소 3배 ~ 15배 느림/ cpu 연산량 자체가 너무 많음. 무거움

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

from sort import * # 객체를 추적한다.

import os
import pandas as pd
import playsound
import threading
from dotenv import load_dotenv, find_dotenv
import pymysql
import boto3


# 종혁 추가 : global 변수
alarm_path = '/Users/jongya/Desktop/Workspace/lab/20230210_sesac_final/yolo7-codingbug/yolov7/beep.wav'
userId = "yoon" # 사용자 id

# 종혁 : RDS, S3 관련 정보
load_dotenv([x for x in os.listdir(os.getcwd()) if x.endswith('.env')][0])
rds_host = os.environ['RDS_HOST']
rds_port = int(os.environ['RDS_PORT'])
rds_database = os.environ['RDS_DATABASE']
rds_username = os.environ['RDS_USERNAME']
rds_password = os.environ['RDS_PASSWORD']
s3_resource = os.environ['S3_RESOURCE']
s3_bucket_name = os.environ['S3_BUCKET_NAME']

# ???
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 종혁 : 로그 저장 -> 변경사항 있음
def saving_logs(txt_path, s, now, det, names):
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        with open(txt_path + '/' + now.strftime("%Y-%m-%d %H-%M-%S.%f") + '.txt', 'w') as f:
            f.write(now.strftime("%Y-%m-%d %H-%M-%S.%f") + s + '\n')

# 이미지 저장
def saving_images(img_path, time_stamp_txt, im0):
    if os.path.exists(img_path):
        pass
    else:
        os.makedirs(img_path)
    
    cv2.imwrite(img_path + '/' + time_stamp_txt + '.jpg', im0)
    
# 종혁 : 알람 사운드
def sounding(soundpath):
    playsound.playsound(soundpath)

# 종혁 : 알람 사운드 멀티 스레드
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
        
# 종혁 : 로그 데이터프레임
def make_log_dataframe(path, event_name):
    log_list = os.listdir(path)
    log_list = [path + '/' + log for log in log_list if log.endswith('.txt')]
    log_list.sort()
    result = []
    global userId
    
    origin_col_name = ['event_name', 'userId', 'detect_date', 'detect_time', 'img_size',
                       'object1_num', 'object1_name',
                       'object2_num', 'object2_name',
                       'object3_num', 'object3_name',]
    
    final_col_name = ['event_name', 'userId', 'detect_date', 'detect_time',
                         'object1_name', 'object1_num',
                         'object2_name', 'object2_num',
                         'object3_name', 'object3_num',]
    
    for log in log_list:
        f = open(log, 'r')
        lines = f.readlines()
        
        for line in lines:
            line = line.replace('\n', '').replace(',', '').replace(':', '').split(' ')
            line.insert(0, userId)
            line.insert(0, event_name)
            
            del line[len(line) - 1]
            
            while len(line) < len(origin_col_name):
                line.append('')
            
            result.append(line)
    log_dataframe = pd.DataFrame(result, columns = origin_col_name)
    log_dataframe = log_dataframe[final_col_name]
    log_dataframe.to_csv(path + f'/logs_{event_name}.csv')
    print("log를 데이터프레임으로 변환 후 저장하였습니다.")
    return log_dataframe
    
# 종혁 : RDS로 로그 정보 올리기
def to_rds(df, host, port, username, database, password):
    
    conn = pymysql.connect(host = host, user = username, port = port,
                           database = database, password = password)

    cursor = conn.cursor()

    sql = 'INSERT INTO test VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
    
    for i in range(len(df)):
        event_name = df.iloc[i]['event_name']
        userId = df.iloc[i]['userId']
        detect_date = df.iloc[i]['detect_date']
        detect_time = df.iloc[i]['detect_time']
        object1_name = df.iloc[i]['object1_name']
        object1_num = df.iloc[i]['object1_num']
        object2_name = df.iloc[i]['object2_name']
        object2_num = df.iloc[i]['object2_num']
        object3_name = df.iloc[i]['object3_name']
        object3_num = df.iloc[i]['object3_num']
        cursor.execute(sql,
                   (event_name, userId, detect_date, detect_time,
                    object1_name, object1_num,
                    object2_name, object2_num,
                    object3_name, object3_num))
    
    conn.commit()
    conn.close()
    print('RDS 전송 완료 : to_rds')
    

# 종혁 : to_rds 멀티 스레드 실행 메서드
def run_to_rds(df, host, port, username, database, password):
    print('RDS 전송 시작 : run_to_rds')
    t = threading.Thread(target = to_rds, args=(df, host, port, username, database, password))
    t.start()
    print('스레드 시동 완료 : run_to_rds')
    

# 종혁 : S3로 이미지 업로드
def to_s3(resource, bucket_name, event_name, path):
    s3 = boto3.resource(resource)
    
    local_file_list = os.listdir(path)
    upload_file_list = [event_name + '/' + x for x in local_file_list]
    local_file_list = [path + '/' +  x for x in local_file_list]
    
    for i in range(len(local_file_list)):
        s3.meta.client.upload_file(local_file_list[i], bucket_name, upload_file_list[i])
    
    s3.meta.client.close()
    print('S3 전송 완료 : to_s3')
    

# 종혁 : to_s3 멀티 스레드 실행 메서드
def run_to_s3(resource, bucket_name, event_name, path):
    print('S3 전송 시작 : run_to_s3')
    t = threading.Thread(target = to_s3, args=(resource, bucket_name, event_name, path))
    t.start()
    print('스레드 시동 완료 : run_to_s3')
    
# 종혁 : gif 만들기
# def make_gif(path, event_name, duration):
#     # path : 이벤트 폴더
#     img_list = os.listdir(path)
#     img_list = [path + '/' + x for x in img_list]
#     images = []
#     for img in img_list:
#         images.append(imageio.imread(img))
        
#     imageio.mimsave(path + event_name + '.gif', images, 'GIF', duration = duration)
    # duration : 프레임 간 전환 속도. 초 단위    

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

def track_track(det, colors, names, im0, img):
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
    dets_to_sort = np.empty((0,6))
    # NOTE: We send in detected object class too
    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
        dets_to_sort = np.vstack((dets_to_sort, 
                    np.array([x1, y1, x2, y2, conf, detclass])))
        print("x1, y1, x2, y2 : ", x1, y1, x2, y2)
        
    if opt.track:

        tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color, opt.remove_old_track) # 종혁 : sort_tracker = sort 임
        tracks =sort_tracker.getTrackers()
        sort_tracker.color_list

        # draw boxes for visualization
        # if len(tracked_dets)>0:
        bbox_xyxy = tracked_dets[:,:4]
        try:
            identities = tracked_dets[:, 8]
        except:
            identities = 'x'
        categories = tracked_dets[:, 4]
        confidences = None

        if opt.show_track:
            #loop over tracks
            for t, track in enumerate(tracks):
    
                track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                [cv2.arrowedLine(im0, (int(track.bottom_center[i][0]),
                                int(track.bottom_center[i][1])), 
                                (int(track.bottom_center[i+1][0]),
                int(track.bottom_center[i+1][1])),
                track_color, thickness=opt.thickness) 
                for i,_ in  enumerate(track.bottom_center)
                if i < len(track.bottom_center)-1 ] 
        else:
            bbox_xyxy = dets_to_sort[:,:4]
            identities = None
            categories = dets_to_sort[:, 5]
            confidences = dets_to_sort[:, 4]
        
        im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
        return im0


def detect(save_img=True):
    detecting_time = opt.detecting_time
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
    remove_old_track = opt.remove_old_track # 종혁 : 추가 - 객체가 사라지면 트랙 삭제

    # 종혁 : event type 세팅
    event_type = 0
    
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

    # for 문 전 선언
    startTime = 0
    event_type = 0
    
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
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # 경로
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det) == 0:
                now = datetime.now()
                time_stamp = now.timestamp()
                time_stamp_txt = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
                if event_type == 0:
                    pass
                
                else:
                    if (time_stamp - time_stamp_old) <= detecting_time: # 객체 사라짐, dt 이하
                        event_type = 2
                        print(f'event_type : {event_type}')
                        
                        im0 = track_track(det, colors, names, im0, img)
                        
                        saving_logs(txt_path, s, now, det, names)
                    
                    elif (time_stamp - time_stamp_old) > detecting_time: # 객체 사라짐, dt 
                        event_type = 0
                        print(f'event_type : {event_type}')
                        
                        im0 = track_track(det, colors, names, im0, img)
                        saving_images(img_path, time_stamp_txt, im0)
                        
                        print('시간차 : ', time_stamp - time_stamp_old)
                        
                        log_dataframe = make_log_dataframe(txt_path, event_name)
                        # run_to_rds(log_dataframe, rds_host, rds_port, rds_username, rds_database, rds_password)
                        
                        # run_to_s3(s3_resource, s3_bucket_name, event_name, img_path)
                        
                        opt.remove_old_track = True
                        im0 = track_track(det, colors, names, im0, img)
                
            elif len(det):
                now = datetime.now()
                time_stamp = now.timestamp()
                time_stamp_txt = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
                if event_type >= 1: # 지속 탐지
                    event_type = 1
                    print(f'event_type : {event_type}')
                    
                    # 로그 저장
                    saving_logs(txt_path, s, now, det, names)

                    # 트래킹 및 이미지 저장
                    im0 = track_track(det, colors, names, im0, img)
                    saving_images(img_path, time_stamp_txt, im0)
                    
                    # 경보 알람
                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        object_bot_center = [int((x1 + x2)/2), int(y2)]
                        danger_range_ratio = opt.danger_range_ratio
                        danger_range = int(im0.shape[0] * danger_range_ratio)
                        if (object_bot_center[1] >= danger_range)&(time_stamp - first_time <= detecting_time):
                            run_sounding(alarm_path)
                    
                    # 타임 스탬프
                    time_stamp_old = time_stamp
                
                elif event_type == 0: # 최초 탐지
                    event_type = 1
                    event_name = now.strftime("%Y-%m-%d-%H-%M-%S")
                    print(f'event_type : {event_type}')
                    
                    # 폴더 생성
                    if not os.path.exists(str(save_dir / event_name)):
                        img_path = str(save_dir / event_name) + '/' + 'img'
                        txt_path = str(save_dir / event_name) + '/' + 'txt'
                        vid_path = str(save_dir) + '/' + event_name
                        os.makedirs(img_path)
                        os.makedirs(txt_path)
                    
                    # 로그 저장
                    saving_logs(txt_path, s, now, det, names)
                    
                    # 트래킹 및 이미지 저장
                    opt.remove_old_track = False
                    im0 = track_track(det, colors, names, im0, img)
                    saving_images(img_path, time_stamp_txt, im0)
                    
                    # 경보 알람
                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        object_bot_center = [int((x1 + x2)/2), int(y2)]
                        danger_range_ratio = opt.danger_range_ratio
                        danger_range = int(im0.shape[0] * danger_range_ratio)
                        if object_bot_center[1] >= danger_range:
                            run_sounding(alarm_path)
                    
                    # 타임 스탬프
                    time_stamp_old = time_stamp
                    first_time = now.timestamp()
                     
            # 종혁 추가
            t4 = time_synchronized()
            
            # if len(det):
            #     # Rescale boxes from img_size to im0 size
            #     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            #     # Print results
            #     for c in det[:, -1].unique():
            #         now = datetime.now()
            #         event_name = now.strftime("%Y-%m-%d-%H-%M-%S.%f")
            #         timestamp = now.strftime("%Y-%m-%d %H-%M-%S.%f")
            #         n = (det[:, -1] == c).sum()  # detections per class
            #         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            #         saving_logs(txt_path, event_name, s, timestamp)
            #         make_log_dataframe(txt_path, event_name)
            
            # # 종혁 : sort tracking
            # # sort 가 내보내는 것은 im0 (이미지), x1, y1, x2, y2 값이어야 한다. -> 이걸 나중에 이용할 수 있어야
            # im0 = track_track(det, colors, names, im0)
            
            
            # Print time (inference + NMS)
            # 종혁 추가 : tracking 쪽 시간측정도 추가 (t4-t3)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(t4 - t3):.1f}ms) operation')


            # Stream results
            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
                

            if view_img:
                # 종혁 추가 : 화면에 danger_range 가 나오도록
                danger_range_ratio = 1.0 - opt.danger_range_ratio
                im0 = make_danger_range_display(danger_range_ratio, im0)
                
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #         print(f" The image with the result is saved in: {save_path}")
            #     else:  # 'video' or 'stream'
            #         if len(det) == 0:  # 탐지된 객체가 없으면
            #             if vid_path is not None:  # 이전에 저장된 비디오 파일이 있으면 저장
            #                 vid_writer.release()
            #                 print(f"The video file {vid_path} is saved.")
            #                 vid_path = None

            #         else:  # 새로운 비디오 파일 생성
            #             vid_writer, vid_path = saving_videos(vid_writer, vid_path, save_dir, im0, vid_cap)

            ## 비디오 경로
            # vid_path = str(save_dir) + '/' + 'temp'
            # ##

            # # 종료 조건 추가
            # if len(det) == 0 and dataset.mode != 'image':
            #     if webcam:
            #         continue
            #     else:
            #         if vid_path is not None:
            #             vid_writer.release()
            #             print(f"The video file {vid_path} is saved.")
            #         cv2.waitKey(0)
            #         break

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
    parser.add_argument('--remove-old-track', action='store_true', help='객체가 사라지면 트랙을 삭제') # 종혁 추가
    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')
    parser.add_argument('--danger-range-ratio', type=float, default=0.5, help='danger range ratio, [0.0 -1.0]') # 종혁 추가 : danger range ratio 받아오기
    parser.add_argument('--detecting-time', type=int, default=10, help='undetected time limit')

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
