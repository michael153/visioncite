#!/bin/bash

# Unpack Python
tar -xzf python.tar.gz
export PATH=$(pwd)/python/bin:$PATH
mkdir home
export HOME=$(pwd)/home

# Train model
cd visioncite
python train.py
cd ..

# Mail results
mail.py "Training Results" "$(cat train.log)" bveeramani@berkeley.edu
mail.py "Training Results" "$(cat train.log)" m.wan@berkeley.edu

# Export model
tar -czvf output.tar.gz ./
