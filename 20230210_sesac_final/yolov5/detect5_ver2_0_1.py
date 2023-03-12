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

Event_type
0 : 객체가 탐지되지 않는 상시 상태
1 : 객체가 탐지됨
2 : 객체가 사라짐


저장 알고리즘(조건문)
if 객체가 없다가 탐지되면
    -> 이벤트 폴더 생성
    -> event_type을 1로 바꾸고
    -> 로그 및 이미지 저장
elif 객체가 탐지될 경우
    -> event_type을 1로 유지하고
    -> 로그 및 이미지 저장
elif 객체가 사라졌고, 사라진지 10초 미만인 경우
    -> event_type을 2로 바꾸고
    -> 로그 및 이미지 저장
elif 객체가 사라졌고, 사라진지 10초 이상인 경우
    -> event_type을 0으로 바꾸고 종료


############################################################################
############################### 업데이트노트 ##################################
                   
%% 성능 개선
- 로그 및 이미지 저장 프로세스 명시
- 기존의 폴더 생성, 결과 저장 프로세스를 현재 프로젝트에 알맞게 조정

%% To Do
- 코드 리팩토링
- 저장 조건 알고리즘 효율화
(1) try except 보다는 if문이 리소스 소모가 적을 것으로 판단됨
(2) if문 사이에서도 순서 재배치를 통한 리소스 소모 최소화를 노릴 수 있음

"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import datetime # 종혁 : 타임스탬프용



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



def save_log_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                    save_img, save_crop, view_img, hide_labels, hide_conf, names,
                    annotator, imc, save_dir, p):
    
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


def stream_results(annotator, view_img, p, windows):        
    im0 = annotator.result()
    if view_img:
        if platform.system() == 'Linux' and p not in windows:
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond


def save_img_method(now, im0, img_path):
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f") # 종혁 추가 : 타임스탬프용 / 타임스탬프 단위가 1초여서 1초 단위로 저장됨
    cv2.imwrite(img_path + '/' + timestamp + '.png', im0) # 종혁 : 이미지 저장


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
    ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    ############### 종혁 : 디렉토리 생성 ###############
    ###############################################
    
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # (save_dir / 'images' if saving_img else save_dir).mkdir(parents=True, exist_ok=True) # 종혁 추가 : 디렉토리 만들기
    
    ###############################################
    ###############################################
    
    ############### 종혁 : 초기 이벤트 설정값 ###############
    ###################################################
    
    event_type = 0 # 상시상태
    
    ###################################################
    ###################################################
    
    
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

            ############### 종혁 : 이벤트 디렉토리 생성 ###############
            #####################################################
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels')  # 종혁 : 텍스트 저장 경로 수정
            img_path = str(save_dir / 'images')  # 종혁 : 이미지 저장 경로
            
            #####################################################
            #####################################################
            
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            
            ############### 종혁 : 로그 데이터 저장 ###############
            ##################################################
            
            now = datetime.datetime.now()
            time_stamp = now.timestamp()
            
            try:
                if (len(det) == 0)&(event_type == 0): # detect 감지 대상이 없는 상시 상태
                    pass
                
                elif len(det)&(event_type == 0): # 처음으로 감지 대상이 잡혔을 때 -> 가장 마지막으로 가는 게 자원효율 상 좋을 것
                    event_type = 1 # 객체 감지됨
                    event_name = now.strftime("%Y-%m-%d %H-%M-%S") # 이벤트명 : 최초탐지시간
                    time_stamp_old = time_stamp # 이전 감지시간 기록
                    txt_path = str(save_dir / event_name / 'logs') # 로그 저장 경로 설정
                    img_path = str(save_dir / event_name / 'images') # 이미지 저장 경로 설정
                    (os.makedirs(txt_path) if save_txt else save_dir.mkdir(parents = True, exist_ok = True)) # 이벤트 폴더 및 로그 폴더 생성
                    (os.makedirs(img_path) if saving_img else save_dir.mkdir(parents = True, exist_ok = True))  # 이미지 폴더 생성
                    
                    if save_txt:
                        save_log_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                                        save_img, save_crop, view_img, hide_labels, hide_conf, names,
                                        annotator, imc, save_dir, p) # 로그 저장
                    
                    if saving_img: # 종혁 추가 : 이미지 세이브할 경우
                        stream_results(annotator, view_img, p, windows) # 이미지 변환
                        save_img_method(now, im0, img_path) # 이미지 저장
                        
                    
                elif len(det)&(event_type == 1): # 이어서 객체가 계속 탐지될 때
                    event_type = 1 # 객체 탐지됨
                    time_stamp_old = time_stamp
                    
                    if save_txt:
                        save_log_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                                        save_img, save_crop, view_img, hide_labels, hide_conf, names,
                                        annotator, imc, save_dir, p) # 로그 저장
                    
                    if saving_img: # 종혁 추가 : 이미지 세이브할 경우
                        stream_results(annotator, view_img, p, windows) # 이미지 변환
                        save_img_method(now, im0, img_path) # 이미지 저장
                    
                elif (len(det) != 1)&(event_type >= 1)&((time_stamp - time_stamp_old) < 10): # 객체가 사라졌고, 사라진지 10초 미만
                    event_type = 2 # 객체 사라짐
                    
                    if save_txt:
                        save_log_method(det, im, im0, s, now, save_txt, gn, save_conf, txt_path,
                                        save_img, save_crop, view_img, hide_labels, hide_conf, names,
                                        annotator, imc, save_dir, p) # 로그 저장
                    
                    if saving_img: # 종혁 추가 : 이미지 세이브할 경우
                        stream_results(annotator, view_img, p, windows) # 이미지 변환
                        save_img_method(now, im0, img_path) # 이미지 저장
                    
                elif (len(det) != 1)&(event_type >= 1)&((time_stamp - time_stamp_old) >= 10): # 객체가 사라졌고, 사라진지 10초 이상
                    event_type = 0 # 상시상태로 전환

            except:
                pass
            
            ##################################################
            ##################################################

            # Stream results
            stream_results(annotator, view_img, p, windows)
            
            
                    
            

            # Save results (image with detections)
            ############### 종혁 : 이미지 데이터 저장 ###############
            ###################################################
            
            try:
                if len(det): # 탐지 객체가 1개 이상일 때
                    temp = 1 # 종혁 : 저장할지 말지 여부
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            for *xyxy, conf, cls in reversed(det):
                                if vid_path[i] != save_path:  # new video
                                    vid_path[i] = save_path
                                    if isinstance(vid_writer[i], cv2.VideoWriter):
                                        vid_writer[i].release()  # release previous video writer
                                    if vid_cap:  # video
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                vid_writer[i].write(im0)
                elif temp == 1:
                    if saving_img: # 종혁 추가 : 이미지 세이브할 경우
                        cv2.imwrite(img_path + '/' + timestamp + '.png', im0) # 종혁 : 이미지 저장
                    temp = 0
            except:
                pass
            
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
