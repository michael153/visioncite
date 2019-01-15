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
import pipeline
from semantic_segmentation import generate_otsu, generate_segments

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

def test_on_data(X, Y, model):
	assert len(X) == len(Y), "Size of X ({0}) and Y ({1}) are unequal".format(len(X), len(Y))
	correct = 0
	total = 0
	probs = model.predict_proba(X)
	for i, y in enumerate(Y):
		if y == 1 and probs[i] > 0.5:
			correct += 1
		if y == 0 and probs[i] < 0.5:
			correct += 1
		total += 1
		print("{0}\t|{1}\t".format(str(probs[i]), y))
	print("\n")
	print("{0}/{1} overall accuracy = {2}%".format(correct, total, 100*correct/total))
	return (correct, total)

def test_on_image(img_file_name, poly_filter, model):
	image = Image.open(os.path.join(images.RAW_IMAGES_PATH, '{0}.jpg'.format(img_file_name)))
	fig, ax = plt.subplots(1)
	generate_otsu(img_file_name)
	segments = generate_segments(
					img_file_name,
					save_segments=False,
					save_image=False,
					display_image=False
			   )
	rects = []
	poly_vecs = []
	label_vecs = []
	for i, segment in enumerate(segments):
		rect = pipeline.get_polygon_bounding_rect(segment)
		angles = pipeline.get_polygon_angles(segment)
		poly_vecs.append([
			len(segment),
			min(angles),
			max(angles),
			rect[1],
			rect[2],
			rect[1]*rect[2],
			pipeline.get_polygon_whitespace_score(img_file_name, rect),
			pipeline.get_polygon_shape_signature(segment, rect)
		])
		label_vecs.append(
			pipeline.get_polygon_vector(
				img_file_name,
				[segments[j] for j in range(len(segments)) if j != i],
				rect)
			)
		rects.append(rect)
	polygon_filter = poly_filter.predict_proba(np.array(poly_vecs))
	filtered = []
	for i, poly in enumerate(polygon_filter):
		if poly > 0.2:
			filtered.append(i)

	probs = model.predict_proba(np.array(label_vecs))
	for i in filtered:
		probs[i] = 0
	best = np.argmax(probs)

	if probs[best] < 0.25:
		print("No title found")

	false_positives = [i for i in range(len(probs)) if probs[i] > 0.5 and i != best]
	ax.imshow(image, cmap=plt.cm.gray)
	ax.add_patch(patches.Polygon(list(segments[best]), linewidth=1, edgecolor='springgreen', linestyle='--', facecolor='none'))
	ax.add_patch(patches.Rectangle(rects[best][0], rects[best][1], rects[best][2], linewidth=1, edgecolor='g', facecolor='none'))
	for i, polygon in enumerate(segments):
		if i != best and i not in false_positives and i not in filtered:
			ax.add_patch(patches.Polygon(list(polygon), linewidth=1, edgecolor='dimgray', linestyle='--', facecolor='none'))
	for i in filtered:
		ax.add_patch(patches.Polygon(list(polygon), linewidth=1, edgecolor='red', linestyle='--', facecolor='none'))
	for fp in false_positives:
		ax.add_patch(patches.Polygon(list(segments[fp]), linewidth=1, edgecolor='lightseagreen', linestyle='--', facecolor='none'))
		ax.add_patch(patches.Rectangle(rects[fp][0], rects[fp][1], rects[fp][2], linewidth=1, edgecolor='r', facecolor='none'))

	plt.figure(2)
	plt.plot(probs)
	plt.figure(3)
	plt.plot(polygon_filter)


poly_filter = load_model("poly_filter")
model = load_model("1547438390")
# data = get_data("1547083109_export_v1547432879")
# X, Y = get_x_y(data["labeled"], data["other"])
# test_on_data(X, Y, model)
# 1545789732DKKVU086ZB.jpg
test_on_image('1545789735BOOWMH88S5', poly_filter, model)

plt.show()