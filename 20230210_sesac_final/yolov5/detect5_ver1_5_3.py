# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
                                 

############################################################################
##########################  detect.py 명세서  ################################
############################################################################

## 실행코드 : python detect5_ver1_5_3.py --weights ./runs/best/best_5.pt --source 0 --save-txt --saving-img --detecting-time 3 --conf-thres 0.2


1. 목적
- 객체인식(탐지) 기술을 이용한 서비스 제공을 위한  그 목적으로 한다.
- 이를 위해
- (1) 객체를 인식하는 기술
- (2) 이 결과를 저장하는 기술
- (3) 저장된 결과를 RDS 와 S3 와 같은 데이터베이스에 올리는 기술
- 들의 원활한 구동을 목표로 한다.


2. 정의
- 객체 : 탐지하려는 대상을 의미한다.
- 이벤트 : 탐지 범위에 탐지된 대상의 수가 1 이상인 때부터 더이상 객체가 탐지되지 않을 때 까지를 한 번의 이벤트라 한다.
- 로그 : 이벤트 동안의 탐지 결과를 요약한 문자 정보. 탐지 일시 및 객체 이름, 객체 수 등을 담는다. txt 파일로 저장된 후 DataFrame화 된다.
- 이미지 : 이벤트 동안의 탐지 영상 캡쳐 이미지. jpg로 저장된다.


3. 기능

3-1. 주요 기능
- YOLOv5 모델을 통한 객체 인식(탐지) 기능
- 객체 인식(탐지) 결과를 이미지와 로그데이터로 자동 저장

3-2. 부가 기능
- 저장된 로그데이터를 RDS에 자동 업로드
- 저장된 이미지파일을 S3에 자동 업로드
- 이미지들을 자동으로 gif로 만드는 기능 (구현했으나, 실행되지 않게 막음)
- 객체가 탐지되면 자동 경보 울림

3-3. 기능의 설명 및 제한 사항
- 이미지와 로그 txt 파일은 '이벤트'별 폴더로 묶여 저장됨
- 로그는 탐지 일시, 탐지 객체의 이름과 수가 기록됨
- 로그는 그 외에 탐지된 객체의 bounding box 좌표를 출력할 수 있으나, 현재는 출력되지 않게 제한함
- 이미지는 이벤트 의 첫 이미지(객체가 탐지된 첫 이미지)와 객체가 미탐지된 마지막 이미지가 저장됨
- 이미지는 바운딩박스 없이 저장됨 (출력할 수 있을 것 같은데, 방법을 모름. 일단 오류로 남겨둠.)


4. 이용 기술
- 객체 인식 및 결과 저장 : YOLOv5
- RDS와의 파일 교환 : pymysql
- S3와의 파일 교환 : boto3
- 경보 사운드 : playsound
- 객체 인식과 파일 교환, 경보 사운드의 병렬 처리 : threading


5. 사용법

5-1. 의존성 설치
- cmd 등의 명령 프롬프트를 통해 본 파일이 있는 곳으로 디렉토리 이동한 뒤, 아래 명령어를 실행해주세요.
- pip install -r requirements.txt
** imageio, pymysql, boto3 등 추가 라이브러리도 requirements에 추가해놓음.
** 그 외 필요한 라이브러리가 있었다면 말씀 부탁드립니다.

5-2. AWS CLI 세팅 : AWS 자동저장 기능 사용시에 한함 (detect.py 버전 1.2.0 이상)
- 로컬 PC에 미리 AWS CLI 세팅을 해 둬야 합니다.
- 세팅을 하지 않을 경우 S3로 이미지가 업로드되지 않거나 오류 발생으로 코드 중지될 수 있습니다.
- 직접 Access key 등을 입력하는 것은 보안상 문제가 있으므로 허용하지 않습니다.

5-3. preference 세팅 : AWS 자동저장 기능 사용시에 한함 (detect.py 버전 1.2.0 이상)
- 아래 코드의 import 부 아래 preference를 세팅해주시기 바랍니다.
- preference 에는 AWS와 연결에 필요한 여러 정보들을 설정하는 부분입니다.

5-3. 실행
- cmd 등의 명령 프롬프트를 통해 본 파일이 있는 곳으로 디렉토리 이동한 뒤, 아래 명령어를 실행해주세요.
- python detect5_ver1_5_5.py --weights ./runs/best/best_2.pt --source 0 --save-txt --saving-img --detecting-time 3 --conf-thres 0.2
- 실행 명령어에 대한 설명은 아래 6.세부 사항에서 다룹니다.


6. 기타 세부 사항

6-1. 명령어 옵션
- python detect5_ver1_5_3.py를 실행할 때의 옵션을 소개한다.
- --weights : 객체 탐지에 이용할 weight 파일(가중치 파일)을 지정한다.
- --source : 객체 탐지할 소스에 대한 유형을 지정. 0은 웹캠.
- --save-txt : 로그정보를 기록할지에 대한 옵션. 선언시 True
- --saving-img : 이미지를 저장할지에 대한 옵션. 선언시 True
- --detecting-time : 객체 미탐지시 얼마의 시간 동안 이벤트 종료를 유예할 건지 설정. 단위는 초(sec)
- --conf-thres : 객체일 가능성(confidence)이 얼마 이상일 때 객체로 인식할 것인지. 단위는 확률 (0 ~ 1.0)
               
6-2. Event_type
0 : 객체가 탐지되지 않는 상시 상태
1 : 객체가 탐지됨
2 : 객체가 사라짐
** {1 -> 2 -> 0} = 1event 

6-3. 저장 알고리즘(조건문)
if 객체가 없다가 탐지되면
    -> 이벤트 폴더 생성
    -> event_type을 1로 바꾸고 이벤트 시작
    -> 로그 및 이미지 저장
elif 객체가 탐지될 경우
    -> event_type을 1로 유지하고
    -> 로그 저장
elif 객체가 사라졌고, 사라진지 10초 미만인 경우
    -> event_type을 2로 바꾸고
    -> 로그 저장
elif 객체가 사라졌고, 사라진지 10초 이상인 경우
    -> event_type을 0으로 바꾸고
    -> 로그 및 이미지 저장
    -> 및 이벤트 종료
    
    
7. detect 코드 버전 내역
ver1.0.0 : 객체 탐지시 이미지 및 로그 저장 기능 테스트
ver1.1.0 : 객체 탐지시 이미지 및 로그 저장 정식 구현
ver1.1.1 : 로그 및 이미지 저장 프로세스 최적화
ver1.1.2 : 로그 및 이미지 저장 코드 리펙토링
ver1.2.0 : parse opt 추가 - 저장 여부 선택 가능
ver1.3.0 : gif 만들기 메서드, log의 dataframe화 구현
ver1.4.0 : AWS RDS, S3와 파일 전송 구현, 멀티스레딩 구현
ver1.4.1 : AWS 연결 보안 강화 - env 이용
ver1.5.0 : 객체 탐지시 경보 울림 구현
ver1.5.1 : 로그 규칙 재정의 (공백 포함하지 않도록)
ver1.5.2 : 이미지 저장 규칙 수정 (모든 이미지 저장 -> 첫 탐지 이미지와 퇴장 x초 후 마지막 이미지만 저장)
ver1.5.3 : 버전 수정 (기존 1, 2, 3 -> 1.1.0 ...)


############################################################################
############################### 업데이트노트 ##################################

%% detect5_ver3_1_2.py 주요 사항
- 버전 네이밍 기준 수정
- 기존 1, 2, 3... 버전에서 1.5.3 버전과 같이 버전 1로 묶음

%% 성능개선

%% To Do
- ver1.5.4 : 보안 이슈로 인한 불편사항 최소화 : 보안 통과하지 않더라도 최소한의 기능은 이용 가능하게, 보안사항 입력 간편하게.
- ver1.5.5 : log 및 이미지에 user-uid, cam-name 부여를 통해 실제 서비스처럼 꾸미기
- ver1.5.6 : 로그값 데이터프레임화 -> RDS 전송을 묶어 다른 스레드에서 처리하게끔 효율화
- ver1.6.0 : 퇴치부 알람 객체에 따라 다른 소리가 재생되도록 구현

"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import datetime # 종혁 : 타임스탬프용
import imageio.v2 as imageio # 종혁 : gif 만들기
import pandas as pd # 종혁 : 데이터프레임 만들기
import pymysql # 종혁 : RDS 연결
import boto3 # 종혁 : S3 연결
import threading # 종혁 : 병렬 처리 위한 라이브러리. 메인 스레드가 yolo를 실행할 동안 서브 스레드가 to_rds, to_s3를 실행
from dotenv import load_dotenv, find_dotenv # 종혁 : AWS 연결 암호화를 위한 환경변수 라이브러리
from playsound import playsound # 경보음 재생을 위한 모듈

# preference : AWS 연결 방법
# 두 가지 방법 중 하나만 진행하면 됩니다.
# (1) 환경변수 방법
# 안전한 AWS 연결을 위해 환경변수를 이용한 암호화를 진행했습니다. (+ gitignore)
# 환경변수 방법을 이용하는 경우 별도의 세팅이 필요합니다. (카톡으로 직접 공유받은 경우, 별도 세팅 필요 없음. 단, 이 때 깃허브 업로드 금지)
load_dotenv([x for x in os.listdir(os.getcwd()) if x.endswith('.env')][0])
# find_dotenv()
rds_host = os.environ['RDS_HOST']
rds_port = int(os.environ['RDS_PORT'])
rds_database = os.environ['RDS_DATABASE']
rds_username = os.environ['RDS_USERNAME']
rds_password = os.environ['RDS_PASSWORD']
s3_resource = os.environ['S3_RESOURCE']
s3_bucket_name = os.environ['S3_BUCKET_NAME']
# (2) 직접선언 방법
# 직접선언할 경우 보안상 문제가 있을 수 있으므로 추천하지 않습니다.
# 반드시 코드 운용 후 보안상 문제 있는 부분은 지우고 저장해주세요.
# rds_host = "" # RDS 엔드포인트
# rds_port = 3306 # RDS 포트 번호
# rds_database = "" # RDS에서 이용할 데이터베이스 명 : antifragile
# rds_username = "" # RDS 계정
# rds_password = "" # RDS 계정 비밀번호
# s3_resource = 's3' # boto3를 이용해 접근할 객체 명 : S3
# s3_bucket_name = 'team06-antifragile-s3' # S3 버킷 이름.


# code start
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


# 이미지와 로그 저장하는 메서드 지정
def save_log_and_img_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                            save_img, save_crop, view_img, hide_labels, hide_conf, names,
                            annotator, imc, save_dir, p, windows, img_path, saving_img, event_type):
    
    if save_txt:
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
        # Write results
        for *xyxy, conf, cls in reversed(det):    
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f") # 종혁 추가 : 타임스탬프
                with open(f'{txt_path}/{timestamp}.txt', 'a') as f:
                    f.write(timestamp + s + '\n')

            if save_img or save_crop or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
            if save_crop:
                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    
    # Stream_results
    stream_results(annotator, view_img, p, windows)
    
    # Save_image
    if saving_img:
        save_img_method(now, im0, img_path, event_type)


def save_log_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                    save_img, save_crop, view_img, hide_labels, hide_conf, names,
                    annotator, imc, save_dir, p, windows):
    
    if save_txt:
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
        # Write results
        for *xyxy, conf, cls in reversed(det):    
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f") # 종혁 추가 : 타임스탬프
                with open(f'{txt_path}/{timestamp}.txt', 'a') as f:
                    f.write(timestamp + s + '\n')

            if save_img or save_crop or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
            if save_crop:
                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        
        # Stream_results
        stream_results(annotator, view_img, p, windows)


# 웹캠 등으로 streaming 되는 자료에 대한 처리 메서드
def stream_results(annotator, view_img, p, windows):        
    im0 = annotator.result()
    if view_img:
        if platform.system() == 'Linux' and p not in windows:
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond


# 이미지 저장 메서드
def save_img_method(now, im0, img_path, event_type):
    
    if event_type == 1:
        img_name = "start"
    elif event_type == 0:
        img_name = "end"
        
    # timestamp = now.strftime("%Y-%m-%d-%H:%M:%S.%f") # 종혁 추가 : 타임스탬프용 / 타임스탬프 단위가 1초여서 1초 단위로 저장됨
    cv2.imwrite(img_path + '/' + img_name + '.jpg', im0) # 종혁 : 이미지 저장


# gif 만들기
def make_gif(path, event_name, duration):
    # path : 이벤트 폴더
    img_list = os.listdir(path)
    img_list = [path + '/' + x for x in img_list]
    images = []
    for img in img_list:
        images.append(imageio.imread(img))
        
    imageio.mimsave(path + event_name + '.gif', images, 'GIF', duration = duration)
    # duration : 프레임 간 전환 속도. 초 단위


# 로그 데이터프레임 만들기 : 나중에는 make_log_and_image 메서드에 내장해야 함.
def make_log_dataframe(path, event_name):
    log_list = os.listdir(path)
    log_list = [path + '/' + log for log in log_list]
    log_list.sort()
    result = []
    
    origin_col_name = ['event_name', 'detect_date', 'detect_time', 'img_size',
                       'object1_num', 'object1_name',
                       'object2_num', 'object2_name',
                       'object3_num', 'object3_name',]
    
    censored_col_name = ['event_name', 'detect_date', 'detect_time',
                         'object1_name', 'object1_num',
                         'object2_name', 'object2_num',
                         'object3_name', 'object3_num',]
    
    for log in log_list:
        f = open(log, 'r')
        lines = f.readlines()
        
        for line in lines:
            line = line.replace('\n', '').replace(',','').split(' ')
            line.insert(0, event_name)
            del line[len(line) - 1]
            
            while len(line) < len(origin_col_name):
                line.append('')
            
            result.append(line)
            
    log_dataframe = pd.DataFrame(result, columns = origin_col_name)
    log_dataframe = log_dataframe[censored_col_name]
    print('log 를 DataFrame 으로 변환하였습니다.')
    
    return log_dataframe


# RDS로 로그 정보 올리기
def to_rds(df, host, port, username, database, password):
    
    conn = pymysql.connect(host = host, user = username, port = port,
                           database = database, password = password)

    cursor = conn.cursor()

    sql = 'INSERT INTO test VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'
    
    for i in range(len(df)):
        event_name = df.iloc[i]['event_name']
        detect_date = df.iloc[i]['detect_date']
        detect_time = df.iloc[i]['detect_time']
        object1_name = df.iloc[i]['object1_name']
        object1_num = df.iloc[i]['object1_num']
        object2_name = df.iloc[i]['object2_name']
        object2_num = df.iloc[i]['object2_num']
        object3_name = df.iloc[i]['object3_name']
        object3_num = df.iloc[i]['object3_num']
        cursor.execute(sql,
                   (event_name, detect_date, detect_time,
                    object1_name, object1_num,
                    object2_name, object2_num,
                    object3_name, object3_num))
    
    conn.commit()
    conn.close()
    print('RDS 전송 완료 : to_rds')
    

# to_rds 멀티 스레드 실행 메서드
def run_to_rds(df, host, port, username, database, password):
    print('RDS 전송 시작 : run_to_rds')
    t = threading.Thread(target = to_rds, args=(df, host, port, username, database, password))
    t.start()
    print('스레드 시동 완료 : run_to_rds')
    

# S3로 이미지 업로드
def to_s3(resource, bucket_name, event_name, path):
    s3 = boto3.resource(resource)
    
    local_file_list = os.listdir(path)
    upload_file_list = [event_name + '/' + x for x in local_file_list]
    local_file_list = [path + '/' +  x for x in local_file_list]
    
    for i in range(len(local_file_list)):
        s3.meta.client.upload_file(local_file_list[i], bucket_name, upload_file_list[i])
    
    s3.meta.client.close()
    print('S3 전송 완료 : to_s3')
    

# to_s3 멀티 스레드 실행 메서드
def run_to_s3(resource, bucket_name, event_name, path):
    print('S3 전송 시작 : run_to_s3')
    t = threading.Thread(target = to_s3, args=(resource, bucket_name, event_name, path))
    t.start()
    print('스레드 시동 완료 : run_to_s3')
    

# 경보음 재생 메서드
def sound_alarm(sound_path, ):
    playsound(sound_path)
   
    
# sound_alarm 멀티 스레드 실행 메서드
def run_sound_alarm(sound_path):
    print('알람 재생 시작')
    t = threading.Thread(target = sound_alarm, args = (sound_path, ))
    t.start()
    print('알람 재생 완료')


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        # data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        data=ROOT / 'data/antifragile.yaml', # 종혁 : 안티프레질 커스텀
        imgsz=(640, 640),  # inference size (height, width)
        # conf_thres=0.25,  # confidence threshold
        conf_thres=0.75, # 종혁 : 신뢰 임계값을 0.75로 상향 조정. 75% 이상 신뢰도일 경우에만 해당 객체로 인식.
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results # 종혁 : 항시 기록
        save_txt=True,  # save results to *.txt # 종혁 : 항시 기록
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        saving_img = False, # 종혁 : 각 프레임 이미지 저장 여부
        detecting_time = 10,
    ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # 저장 경로 선언
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 기본 디렉토리 increment run
    
    # 이벤트 타입 선언
    event_type = 0 # 이벤트 타입 : 0 상시상태(객체 미탐지) 1 객체 탐지됨 2 객체 사라짐
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)


            # 세이브 경로 지정 (p : 자동 increment)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            
            
            # 로그값 및 이미지값 기본 변수
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            
            # 타임스탬프 timestamp
            now = datetime.datetime.now()
            time_stamp = now.timestamp()
            
            # 객체 탐지시 로그, 이미지 저장 메서드 실행 부
            if (save_txt)|(saving_img): # 로그나 텍스트를 저장하기로 했다면
                try:
                    if (len(det) == 0)&(event_type == 0): # detect 감지 대상이 없는 상시 상태
                        print(f'event_type : {event_type}')
                        pass
                        
                    elif len(det)&(event_type == 0): # 처음으로 감지 대상이 잡혔을 때 -> refactoring시 가장 마지막으로 가는 게 자원효율 상 좋을 것
                        event_type = 1 # 객체 감지됨
                        event_name = now.strftime("%Y-%m-%d-%H-%M-%S") # 이벤트명 : 최초탐지시간
                        time_stamp_old = time_stamp # 이전 감지시간 기록
                        txt_path = str(save_dir / event_name / 'logs') # 로그 저장 경로 설정
                        img_path = str(save_dir / event_name / 'images') # 이미지 저장 경로 설정
                        (os.makedirs(txt_path) if save_txt else save_dir.mkdir(parents = True, exist_ok = True)) # 이벤트 폴더 및 로그 폴더 생성
                        (os.makedirs(img_path) if saving_img else save_dir.mkdir(parents = True, exist_ok = True))  # 이미지 폴더 생성
                        
                        save_log_and_img_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                                                save_img, save_crop, view_img, hide_labels, hide_conf, names,
                                                annotator, imc, save_dir, p, windows, img_path, saving_img, event_type)
                        print(f'event_type : {event_type}')
                        
                        run_sound_alarm('/Users/jongya/Desktop/Workspace/lab/20230210_sesac_final/yolov5/beep.wav')
                            
                        
                    elif len(det)&(event_type >= 1): # 이어서 객체가 계속 탐지될 때
                        event_type = 1 # 객체 탐지됨
                        time_stamp_old = time_stamp
                        
                        save_log_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                                        save_img, save_crop, view_img, hide_labels, hide_conf, names,
                                        annotator, imc, save_dir, p, windows)
                        print(f'event_type : {event_type}')
                        
                        
                    elif (len(det) != 1)&(event_type >= 1)&((time_stamp - time_stamp_old) < detecting_time): # 객체가 사라졌고, 사라진지 x초 미만
                        event_type = 2 # 객체 사라짐
                        
                        save_log_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                                        save_img, save_crop, view_img, hide_labels, hide_conf, names,
                                        annotator, imc, save_dir, p, windows)
                        print(f'event_type : {event_type}')
                    
                        
                    elif (len(det) != 1)&(event_type >= 1)&((time_stamp - time_stamp_old) >= detecting_time): # 객체가 사라졌고, 사라진지 x초 이상
                        event_type = 0 # 상시상태로 전환
                        print(f'event_type : {event_type}')
                        
                        save_log_and_img_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                                                save_img, save_crop, view_img, hide_labels, hide_conf, names,
                                                annotator, imc, save_dir, p, windows, img_path, saving_img, event_type)
                        
                        # 모든 로그값을 불러와 DataFrame 형태로 convert 하여 저장
                        log_dataframe = make_log_dataframe(txt_path, event_name) # 로그 데이터프레임화
                        log_path = str(save_dir / event_name / f'logs_{event_name}.csv') # 로그 csv 파일 저장 경로
                        log_dataframe.to_csv(log_path) # 로그 데이터프레임 csv 파일로 저장
                        
                        # 이미지를 gif로 convert 하여 저장
                        # make_gif(img_path, event_name, 0.5) # 만들었지만 봉인 (시간 오래걸리고 결과물 용량 큼)
                        
                        # 로그값을 RDS에 업로드
                        run_to_rds(log_dataframe, rds_host, rds_port, rds_username, rds_database, rds_password)
                        
                        # 이미지를 S3에 업로드
                        run_to_s3(s3_resource, s3_bucket_name, event_name, img_path)
                        
                        # To Do. execfile(method_alarm())

                except:
                    pass
            
            # Stream results
            # stream_results(annotator, view_img, p, windows)
            
            
            # Save results (image with detections)
            ############### 종혁 : 영상 저장 ###############
            ###################################################
            
            # try:
            #     if len(det): # 탐지 객체가 1개 이상일 때
            #         temp = 1 # 종혁 : 저장할지 말지 여부
            #         if save_img:
            #             if dataset.mode == 'image':
            #                 cv2.imwrite(save_path, im0)
            #             else:  # 'video' or 'stream'
            #                 for *xyxy, conf, cls in reversed(det):
            #                     if vid_path[i] != save_path:  # new video
            #                         vid_path[i] = save_path
            #                         if isinstance(vid_writer[i], cv2.VideoWriter):
            #                             vid_writer[i].release()  # release previous video writer
            #                         if vid_cap:  # video
            #                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #                         else:  # stream
            #                             fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #                         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #                     vid_writer[i].write(im0)
            #     elif temp == 1:
            #         if saving_img: # 종혁 추가 : 이미지 세이브할 경우
            #             cv2.imwrite(img_path + '/' + timestamp + '.png', im0) # 종혁 : 이미지 저장
            #         temp = 0
            # except:
            #     pass
            
            ###################################################
            ###################################################
                            

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/aintifragile.yaml', help='(optional) dataset.yaml path') # 종혁 : 안티프레질로 기본 설정
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold') # 종혁 : 0.6로 상향
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--saving-img', action='store_true', help='saving img each frame') # 종혁 : 이미지 저장 옵션 추가
    parser.add_argument('--detecting-time', type=int, default=10, help='undetected time limit')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
