sudo apt-get install libmecab-dev
sudo apt-get install mecab mecab-ipadic-utf8

# Check the latest version at https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.3-20170922.tar.gz
tar xzvf mecab-ko-dic-2.0.3-20170922.tar.gz

cd mecab-ko-dic-2.0.3-20170922/
autogen.sh
./configure
make
sudo make install

sudo pip install mecab-python
#sudo pip install mecab-python3
