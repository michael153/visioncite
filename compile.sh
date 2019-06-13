#!/bin/bash
PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
PYTHON_DOWNLOAD_FILENAME=Python-3.6.8.tgz

curl -o $PYTHON_DOWNLOAD_FILENAME $PYTHON_DOWNLOAD_URL

mkdir python
tar -xzf $PYTHON_DOWNLOAD_FILENAME

LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

cd "${PYTHON_DOWNLOAD_FILENAME%.*}"
./configure --prefix=$(pwd)/../python
make
make install

cd ..
ls python
ls python/bin

if [ ! -f ./python/bin/python ]
then
    cp python/bin/python3 python/bin/python
fi

if [ ! -f ./python/bin/pip ]
then
    cp python/bin/pip3 python/bin/pip
fi

export PATH=$(pwd)/python/bin:$PATH
# Instasll dependencies here!
pip install --upgrade pip
pip install pillow
pip install torch torchvision
pip install -U matplotlib
pip install sendgrid

tar -czvf python.tar.gz python/
rm Python-3.6.8.tgz
exit
