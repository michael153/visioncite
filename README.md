# Eyecite

Eyecite is an in-progress tool that allows users to accurately cite webpages.

## Usage
```usage: train.py [-h] [-b BATCH_SIZE] [-e NUM_EPOCHS] [--disable-cuda]
                x_train y_train x_test y_test

Train model

positional arguments:
  x_train               path to train dataset image directory
  y_train               path to train dataset label directory
  x_test                path to validation dataset image directory
  y_test                path to validation dataset label directory

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batches BATCH_SIZE
                        number of samples to propogate (default: 64)
  -e NUM_EPOCHS, --epochs NUM_EPOCHS
                        number of passes through dataset (default: 32)
  --disable-cuda        Disable CUDA```

## Dependencies
- numpy
- pytorch
- PIL
- sendgrid
- matplotlib
