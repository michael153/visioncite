mkdir python
tar -xzf Python-3.6.8.tgz

LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

cd Python-3.6.8
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
pip install --upgrade pip
pip install tensorflow==1.5
pip install keras
pip install pillow
pip install torch torchvision
pip install -U matplotlib
pip install sendgrid
pip install python-dotenv
 
tar -czvf python.tar.gz python/
exit

