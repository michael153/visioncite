# Eyecite

Eyecite is an in-progress tool that allows users to accurately cite webpages.

### Compiling Python
To compile Python on CHTC, run `condor_submit -i compile.sub`. After the
interactive session has started, run `chmod +x compile.sh` and then
`./compile.sh`. Once the script has finished running, enter `exit` to exit the
interactive session. You should see a zipped file named `python.tar.gz` in
your home directory after following these instructions.

### Usage
Download the directories `xtrain`, `ytrain`, `xtest`, `ytest` from the Google
Drive. Then, run the command `condor_submit train.sub`.

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
