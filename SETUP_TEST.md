#HOWTO Install & Test 

README.md에 설치및 사용법이 있으나, 몇가지 불편한점이 있이서 제 환경에 맞게 다시 작성하였습니다.

## Requirements

* Python 2.7
* Theano
* Numpy
* [KoNLPy](http://konlpy-ko.readthedocs.io/ko/v0.4.3/)
* [MeCab](https://bitbucket.org/eunjeon/mecab-ko/overview)
* Flask

## 1. Get Source 
```
git clone https://github.com/jaejunh/KoNER.git
cd KoNER
```

## 2. Install python related + redis server 
```
sudo apt-get install python-virtualenv python-dev libmecab-dev mecab mecab-ipadic-utf8 httpie
sudo apt-get redis-server
sudo /etc/init.d/redis-server restart
```

## 3. Get mecab-ko-dic*.tar.gz 
```
# wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.3-20170922.tar.gz
tar xzvf mecab-ko-dic-2.0.3-20170922.tar.gz
cd mecab-ko-dic-2.0.3-20170922
autogen.sh
./configure
make ; sudo make install  # it will install to /usr/lib/x86_64-linux-gnu/mecab
sudo ln -s /usr/lib/x86_64-linux-gnu/mecab /usr/local/lib
```

## 4. Setup Python virtualenv 
```
virtualenv konerenv
source konerenv/bin/activate

pip install -r requirements.txt

tar xzvfp model.tar.gz
cp data/gazette.orig data/all.gazette 
# or: scp <repository>:/home/embian/jaejunh/data.tgz  . && tar xzvfp data.tgz

mkdir -p testdata
touch testdata/fdist.csv        # word freq file
touch testdata/fdist.csv.veto   # veto file
# or: scp <repository>:/home/embian/jaejunh/testdata.tgz . && tar xzvfp testdata.tgz


```
## 5. Run Server Mode 
```
# It will take about 200sec+ for the first time. Later, it will take 20~30sec.
python ./main.py -s              
# optionally,  for command line mode, read README.md

```
## 6. Test 
```
# ping the server
http http://localhost:5000/
http http://localhost:5000/ --pretty=format

# send json query without session.  simplest form
http POST http://localhost:5000/koner/api/v1.0/tag text='류현진은 LA 다저스야구단의 투수다'

# send json query with common session, 'test1234' 
http POST http://localhost:5000/koner/api/v1.0/tag sessionkey='test1234' text='29일 손흥민이 런던 리버풀FC 구단을 상대로 골을 넣었다'
http POST http://localhost:5000/koner/api/v1.0/tag sessionkey='test1234' text='뮤리뉴 감독은 가슴이 아팠다'

# send json query with all flags on:  type=debug, other=<anything>
http POST http://localhost:5000/koner/api/v1.0/tag other='' type=debug sessionkey='com.iloen.melon' text=' 선미 상세정보 닫기 71 25 연관 아티스트 소속 그룹 유닛 원더걸스 같은 소속사 아티스트 박원 어반자카파 플레이리스트. 불러오기 버튼 곡 목록이 없습니다. '

curl -i -H  "Content-Type: application/json" -X POST -d '{"text":"26일(한국시간) 손흥민은 영국 런던 >웸블리 스타디움에서 열린 2017-2018 영국 프리미어리그 20라운드 사우스햄튼 FC와의 홈경기에서 시즌 9호골 을 터트리며 팀의 5-2 승리에 힘을 보탰다."}' http://localhost:5000/koner/api/v1.0/tag
```

## 7. Update Data Files 

* data/gazette  
* testdata/fdist.csv (optional, word freq file)
* testdata/fdist.csv.veto (optional, veto word list for NER)
