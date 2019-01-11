# Copyright 2018 Balaji Veeramani, Michael Wan
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Author: Michael Wan <m.wan@berkeley.edu>

import os
import os.path
import sys
import time
import json

import numpy as np
import sklearn
import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, model_from_json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

import assets
import images

def get_data(filename):
    return json.load(open(os.path.join(assets.DATA_PATH, 'training/{0}.train'.format(filename)), "r"))

def get_x_y(labeled, regular):
    x = np.array(labeled + regular)
    y = np.array([1]*len(labeled) + [0]*len(regular))
    return x, y

def load_model(epoch_id):
    directory_path = assets.DATA_PATH + '/ml/{0}'.format(epoch_id)
    with open(directory_path + '/model_json', 'r') as f:
        json_data = f.read()
        saved_model = model_from_json(json_data)
        saved_model.load_weights(directory_path + '/weights')
        return saved_model

def test_model(X, Y, model):
	assert len(X) == len(Y), "Size of X ({0}) and Y ({1}) are unequal".format(len(X), len(Y))
	correct = 0
	total = 0
	prob = model.predict_proba(X)
	for i, y in enumerate(Y):
		if y == 1 and prob[i] > 0.5:
			correct += 1
		if y == 0 and prob[i] < 0.5:
			correct += 1
		total += 1
		print("{0}\t|{1}\t".format(str(prob[i]), y))
	print("\n")
	print("{0}/{1} overral accuracy = {2}%".format(correct, total, 100*correct/total))
	return (correct, total)

model = load_model("1547187739")
data = get_data("1547083109_export")
X, Y = get_x_y(data["labeled"], data["other"])
test_model(X, Y, model)
