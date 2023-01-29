# Intro

* 애플 앱스토어의 앱 리뷰를 스크랩합니다.  
* 최근 작성된 500개의 리뷰를 스크랩합니다.  
* 업데이트 : 2023-01-29  


<br><br>

# Environment

이번 실험에서 사용된 환경과 라이브러리는 아래와 같습니다.  

* Python 3.8  
* BeautifulSoup  
* urllib  
* pandas  

<br><br>

# Code init

코드의 상단에서 스크래핑 대상과 최종 자료 파일의 이름을 정의합니다.

```python
# 스크래핑 대상 및 최종 자료 파일 이름 정의

country = 'us'          # 국가 코드
app_id = '1163786766'   # 앱스토워 웹페이지에서 확인 가능
app_name = 'alarmy'     # 아무거나 입력해도 됨
scrap_date = '20230129' # 스크래핑 일자 (달라도 상관 없음)
last_page = 10          # 현재 10페이지까지만 지원됨
file_name = f'scrap_apple_{app_name}_{country}_{scrap_date}.csv' # 저장할 파일 이름

# 국가코드 -- 한국 : kr / 미국 : us ...
# 그 외 국가코드는 https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
```

<br><br>

# 참고

* 앱 코드는 웹상 앱스토어 주소창에서 확인 가능 (id 뒤의 숫자만)  

* 국가코드는 아래 페이지에서 확인 가능 (영문 두글자)  

<br><br>
