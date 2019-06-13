#!/bin/bash

# Settings
BATCH_SIZE=16
NUM_EPOCHS=32

# Unpack Python
tar -xzf python.tar.gz
export PATH=$(pwd)/python/bin:$PATH
mkdir home
export HOME=$(pwd)/home

# Train model
train.py -b $BATCH_SIZE -e $NUM_EPOCHS xtrain ytrain xtest ytest &> train.log

# Mail Results
mail.py "Training Results" "$(cat train.log)" bveeramani@berkeley.edu
mail.py "Training Results" "$(cat train.log)" m.wan@berkeley.edu

# Export model
tar -czvf output.tar.gz ./
