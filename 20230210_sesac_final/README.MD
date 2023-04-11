# COMPUTER VISION 기술을 이용한 유해 야생동물 탐지, 퇴치 및 경로 추적 솔루션

**ANTI-FRAGILE 팀 : 박종혁, 심재훈, 손원준, 윤율**  

<video controls autoplay>
<source src="./02. 서비스 동영상_anti_fragile.mp4">
</video>

## 1. INTRO

* 발표자료 : [발표자료.pdf](/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C_anti-fragile.pdf)  
* 프로젝트 상세내용 : [프로젝트 상세내용.xlsx](/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%EC%83%81%EC%84%B8%EB%82%B4%EC%9A%A9_antifragile.xlsx)  

<br>
<br>

## 2. 사용 준비  

미리 훈련된 weight 파일을 사용하고자 하는 경우 `4.실행` 부분을 참고해주세요.  

### 데이터셋 다운로드 (직접 훈련시)  

* 야생동물 데이터 다운로드  
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=645  

* COCO 데이터셋  
https://public.roboflow.com/object-detection/microsoft-coco-subset  

* KAIST 야간 보행자 데이터셋  
https://soonminhwang.github.io/rgbt-ped-detection/  


### requirements 설치  

```terminal
pip install -r ./yolov7/requirements.txt
```

<br>
<br>

## 3. Train  

모델 훈련의 자세한 사항은 훈련 코드들을 참고해주세요.  

[주야간 통합 훈련 : 야생동물만](/train/yolov7_train_%EC%95%BC%EC%83%9D%EB%8F%99%EB%AC%BC.ipynb)  
[주간 + COCO 데이터셋 훈련](/train/yolov7_train_coco%EA%B0%80%EC%A4%91%EC%B9%98%2B%EC%95%BC%EC%83%9D%EB%8F%99%EB%AC%BC4%EC%A2%85_%EC%A3%BC%EA%B0%84_%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5.ipynb)  
[야간 + COCO 데이터셋 훈련](/train/yolov7_train_coco%EA%B0%80%EC%A4%91%EC%B9%98%2B%EC%95%BC%EC%83%9D%EB%8F%99%EB%AC%BC4%EC%A2%85_%EC%A3%BC%EA%B0%84_%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5.ipynb)  
[야간 + KAIST 데이터셋 훈련](/train/yolov7_train_KAIST%EC%95%BC%EA%B0%84%EB%B3%B4%ED%96%89%EC%9E%90%2B%EC%95%BC%EC%83%9D%EB%8F%99%EB%AC%BC4%EC%A2%85_%EC%95%BC%EA%B0%84_%EC%9D%BC%EB%B0%98%ED%95%99%EC%8A%B5.ipynb)  

<br>
<br>

## 4. 객체 탐지(detect) 코드 실행  

객체 탐지 코드를 실행시켜주세요.  
* 객체 탐지 코드 : `./yolov7/detect_or_trackmbasic_1_3_7.py`  

```terminal
-- 주간 detect 코드
python detect_or_trackmbasic_1_3_7.py --weights coco2wild4_day_31epoch_best.pt --no-trace --view-img --track --show-track --unique-track-color --danger-range 0.5 --conf-thres 0.7 --detecting-time 10 --thickness 3 --source 0 --class 1 2 3

-- 야간 detect 코드
python detect_or_trackmbasic_1_3_7.py --weights coco2wild4_night_12epoch_best.pt --no-trace --view-img --track --show-track --unique-track-color --danger-range 0.5 --conf-thres 0.7 --detecting-time 10 --thickness 3 --source 0 --class 0 1 2

```

```python
--weights : 객체 탐지에 이용할 weight 파일(가중치 파일)을 지정한다.
--source : 객체 탐지할 소스에 대한 유형을 지정. 0은 웹캠.
--save-txt : 로그정보를 기록할지에 대한 옵션. 선언시 True
--saving-img : 이미지를 저장할지에 대한 옵션. 선언시 True
--detecting-time : 객체 미탐지시 얼마의 시간 동안 이벤트 종료를 유예할 건지 설정. 단위는 초(sec)
--remove-old-track : 사라진 객체의 tracking 선을 제거합니다.
--danger-range 0.x : 객체가 화면의 x% 범위 내로 접근하면 알람이 울리도록 범위 설정 [0.0-1.0]
--conf-thres 0.y : 객체 인식 신뢰도 [0.0-1.0]
--view-img : 탐지 현황 영상을 보여준다.
--track : 객체 경로 추적을 활성화 한다.
--show-track : 객체 경로를 탐지 현황 영상에 표시한다.
--unique-track-color : 객체 경로를 서로 다른 색으로 지정한다.
```

