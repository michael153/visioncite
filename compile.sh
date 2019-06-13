#!/bin/bash
PYTHON_URL=https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
PYTHON_DIRECTORY=Python-3.6.8.tgz

curl -o $PYTHON_DIRECTORY $PYTHON_URL

mkdir python
tar -xzf $PYTHON_DIRECTORY

LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

cd $PYTHON_DIRECTORY
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
exit
