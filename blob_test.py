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

def load_model(epoch_id):
    directory_path = assets.DATA_PATH + '/ml/{0}'.format(epoch_id)
    with open(directory_path + '/model_json', 'r') as f:
        json_data = f.read()
        saved_model = model_from_json(json_data)
        saved_model.load_weights(directory_path + '/weights')
        return saved_model

def test_model(url, model):
	blobs = get_blobs_by_url(url, pipeline.image_dict[url], display=True, useSavedData=True)[1]
	datapoints = list(map(pipeline.blob_to_trainable, blobs))
	im = Image.open(images.RAW_IMAGES_PATH + '/{0}.jpg'.format(pipeline.image_dict[url]))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(im)
	for i, blob in enumerate(blobs):
		words = blobs[1]
		font = blobs[2]
		datapoint = datapoints[i]
		x1 = datapoint[0]
		x2 = datapoint[1]
		y1 = datapoint[2]
		y2 = datapoint[3]
		prob = model.predict_proba(np.array([datapoint]))[0]
		print(prob)
		if prob > 0.25:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fc = 'none', ec = 'green')
		else:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fc = 'none', ec = 'red')
		ax.add_patch(p)
	plt.show()

model = load_model("1545696096")
test_model("http://www.abc.net.au/local/videos/2008/11/26/2430325.htm", model)
