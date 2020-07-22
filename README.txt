기본 사용법
> conda activate for_django 
> python manage.py runserver


20/6/14 개발일기
- Django tutorial 뽀갬(https://tutorial.djangogirls.org/ko/) : 예제로 개인 블로그 만들었음
- 로그인 / 게시글 작성 / 수정 / 삭제 / 임시저장되었던 게시글 포스팅이 가능
- 무료 웹호스팅(pythonanythere.com) 하고 있음. : MEGO계정을 따로 하나 만들어서 호스팅하면 프로토타이핑에는 문제가 없을 것 같음.(아마도. but max 용량 512MB로 용량 초과를 고려한 방안 생각할 필요는 있음)

- To Do Next
1) multi-user with independent data collection(서로 다른 유저가 각자의 데이터에 침범하지 않게 하나의 model(data structure)에 여러 instance가 동시에 존재할 수 있도록 만들기)
2) ID 생성하는 거 웹에서 할 수 있도록 하기
3) 웹에서 어떻게 circle of life 띄울 지 생각하기
4) 예쁜 디자인을 찾기
5) MEgo 홈페이지 만들기

- Github : https://github.com/JECH2/my-first-blog.git
- Webpage : http://eunjin.pythonanywhere.com/

< Additional Information >
1) Django를 별도의 가상환경(나는 conda를 이용하여 만듬)에 설치해야 한다.
example : conda create -n for_Django python=3.7 django
2) Django로 만든 서버를 로컬에서 실행하기 위해서는 프로젝트의 가장 상위 폴더에 위치한 manage.py를 실행한다.
example : python manage.py runserver
이렇게 하면 127.0.0.1:8000 을 통해 접근이 가능하다.
3) 주로 수정하는 건
urls.py : 블로그에서 접근 가능한 페이지 url를 명시
views.py : 각 url에 대한 리퀘스트가 들어오면 그와 관련된 html 파일을 연결
models.py : 일종의 데이더베이스 구조를 명시하는 부분.
static/css : css 파일을 저장하는 곳
templates/blog or registration : html 파일을 저장하는 곳