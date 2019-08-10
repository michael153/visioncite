#!/bin/bash

# Unpack Python
tar -xzf python.tar.gz
export PATH=$(pwd)/python/bin:$PATH
mkdir home
export HOME=$(pwd)/home

# Train model
python train.py &> train.tb
# Print to Condor log
cat train.tb

# Mail results
python mail.py --subject "Training Results" --message "$(cat train.tb)" bveeramani@berkeley.edu
python mail.py --subject "Training Results" --message "$(cat train.tb)" m.wan@berkeley.edu
rm train.tb
