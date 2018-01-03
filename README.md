# KoNER

KoNER (Korean Named Entity Recognizer)는 한국어 개체명 인식기이다. 

## Requirements

* Python 2.7
* Theano
* Numpy
* [KoNLPy](http://konlpy-ko.readthedocs.io/ko/v0.4.3/)
* [MeCab](https://bitbucket.org/eunjeon/mecab-ko/overview) 

## Installation
* Mecab installation
```
>> sudo bash install_mecab.sh
```
* Other libraries (theano, numpy, konlpy)
```
>> pip install -r requirements.txt
```

## Pre-trained Model
```
>> tar -xvf model.tar.gz
```

## Run Command
```
python main.py -i [input_file_path] -o [output_file_path] -m [model_path] -p [input type 0(=raw), 1(=pos)]
```
* Example
```
>> python main.py -i './data/test_input.txt' -o './data/test_result.txt' -m './model' -p 0
```
## NEW Command for flask web service
```
python main.py -s  
``

Then, try:
```
curl -i -H  -H "Content-Type: application/json" -X POST -d '{"text":"26일(한국시간) 손흥민은 영국 런던 웸블리 스타디움에서 열린 2017-2018 영국 프리미어리그 20라운드 사우스햄튼 FC와의 홈경기에서 시즌 9호골을 터트리며 팀의 5-2 승리
에 힘을 보탰다."}' http://localhost:5000/korner/api/v1.0/tag
```
