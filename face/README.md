# Face Detection (by openCV)
개소리 없이 바로 코드 설명에 들어간다. 
``` python
import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread("../../pictures/people.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv.imshow('img', img)
cv.waitKey(0)
cv.destoryAllWindows()
```
이게 코드이다. 위에서 부터 이 놈은 뭐하는 놈인지 설명을 하겠다. <br/> <br/> <br/>
Line 1 ~ 5
``` python
import numpy as np
import cv2 as cv


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
```
라이브러리를 불러온다. 이때 불러오는 라이브러리는 <a href="https://numpy.org/"> NumPy </a> 와 <a href="https://opencv.org/"> OpenCV </a> 라이브러리를 불러온다. <br/> <br/>
이 둘은 최대한 짧게 설명하고 가자면 Numpy는 C언어로 구성된 Python 라이브러리로 고-오 성능의 계산을 위해서 만들어진 라이브러리 이다. openCV는 컴퓨터 비전을 목적으로 한 인텔에서 개발한 실시간 이미지 프로세싱에 이점을 둔 라이브러리 이다. 
<br/> <br/>

<a href="https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml"> haarcascade_frontalface_default.xml </a> 와 <a href="https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml">haarcascade_eye.xml </a>, 이 둘은 사람의 얼굴과 눈을 찾을 때 도와주는(?) 일종의 공부 데이터라고 생각하면 쉬울거 같다..  2001년에 마이클 존스(Michael Jones), 폴 비올라(Paul Viola) 개발자가 “Rapid Object Detection using a Boosted Cascade of Simple Features” 제안한 방식이다. 사실 나도 이건 잘 이해가 안된다... 정 궁금하다면 <a herf="https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html"> 이 문서 </a> 를 읽어보길 바란다. <br/>  <br/>  
Line 7~8
``` python
img = cv.imread("../../pictures/people.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
``` 
가장 먼저 openCV 라이브러리를 통하여 사진을 불러온다. cv.cvtColor는 opencv의 함수 중 하나로 색을 변경하는 함수이다. 쓰는 방법은 cvtColor(사진, 색) 이렇게 쓰면 된다. 색을 쓸때에는 COLOR_BGR2GRAY, COLOR_RGB2GRAY 등등 여러가지가 있으니 자세한 내용은 <a href="https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html"> 이 문서 </a>를 확인하길 바란다. 

<br/>
Line 10~17

```python
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
```
어...detectMultiScale 이 함수 설명부터 해야할거 같다. detectMultiScale는 입력받은 이미지에서 크기가 다른 객체를 검출 하는 함수라고 생각하면 된다. 앞에 face_cascade를 붙이거 처럼 어떤 객체를 검출을 할지 정해준다. 그럼 detectMultiScale이 함수 안에는 무엇을 적는 것일까? 사실 인자값이 7개나 들어가 맞는데 그럴 필요는 또 없나보다... ~~뭔소리지 진짜로~~ <br/>
아무튼 얼굴의 갯수를 세어준다. <br/>

얼굴의 갯수만큼 for루프를 돈다. 루프를 돌때 가장 먼저 사각형을 그린다. 이 사각형은 우리가 openCV에서 찾은 얼굴을 시각화 시키기 위해서 그리는 것이다. 사각형을 그릴때에는 rectangle() 함수를 활용한다. 이 함수에 대한 예제는 상위 디렉토리에 있는 Basic of Basic 에 있다. 자세한 설명은 늘 그렇듯이 <a href="https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html"> 공식 문서 </a> ~~영어 공부 열심히 하자....~~ <br/>

roi_gray, 위에 있던 gray 하고는 차이가 있다. 변수 명을 통해서 알수 있듯 일단 roi가 붙어 있다. 그럼 roi는 무엇일까? Region of Interest(관심 영역 자르기)의 약자라고 한다. 솔직하게 말해서 roi_gray, roi_color 이거 잘 이해가 안된다... 그리고 설명이나 코멘트가 되어 있는 문서가 별로 없어서 설명하기가 어렵다... ~~죄송합니다 이건 패스..~~ <br/>

eyes는 위에 있던 faces 변수와 같은 부류? 라고 생각하면 될 것 같다. detectMultiScale 함수를 활용하여 사람의 눈을 확인하는 것이다. 여기서 왜 이렇게 확인을 하나 싶을수도 있다. 음 공식 문서에 원문 그대로 인용을 하자면 <b> <l> 눈은 항상 얼굴에 있기 때문에 !!! </b> </l> 라고 한다. 얼굴에 눈이 항상 있기 때문에 한번 더 검출을 해주는 것 이다. 눈 까지 찾았으면 눈에도 사각형을 그려준다. <br/> <br/>

Line 19~21

```python 
cv.imshow('img', img)
cv.waitKey(0)
cv.destoryAllWindows()
```
결과 값, 사람의 얼굴을 찾은 결과를 아까 사진에 그린 사각형을 보여주는 것이다. 따로 저장은 하지 않고 보여주기만 한다. waitKey는 아무키나 기다리는 것이다. 그리고 키가 입력 되면 cv.destoryAllWindows() 이 함수로 인해 모든 창이 사라지고 끝난다. 