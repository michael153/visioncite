# Eyecite

Eyecite is an in-progress tool that allows users to accurately cite webpages.

### Usage
```
usage: train.py [-h] [--xtest XTEST] [--ytest YTEST] [--batches BATCH_SIZE]
                [--epochs NUM_EPOCHS] [--disable-cuda]
                xtrain ytrain

Train model

positional arguments:
  xtrain                path to train dataset image directory
  ytrain                path to train dataset label directory

optional arguments:
  -h, --help            show this help message and exit
  --xtest XTEST         path to validation dataset image directory
  --ytest YTEST         path to validation dataset label directory
  --batches BATCH_SIZE  number of samples to propogate (default: 64)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 32)
  --disable-cuda        disable CUDA support
```

### Dependencies
- numpy
- pytorch
- PIL
- sendgrid
- matplotlib
