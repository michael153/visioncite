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
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import assets
import images
import pipeline

def read_blob_file(file_name):
    return json.load(open(assets.DATA_PATH + "/web_generated/" + file_name + ".json", "r"))

def gather_blobs(blob_data):
    blobs = []
    for picture_id in blob_data:
        font = blob_data[picture_id]["font"]
        locs = blob_data[picture_id]["locs"]
        words = blob_data[picture_id]["words"]
        assert len(font) == len(locs) == len(words), 'Improper blob data formatting: len(f,l,w) = {0}'.format((len(font), len(locs), len(words)))
        for i in range(len(font)):
            loc = [locs[i]["x1"], locs[i]["x2"], locs[i]["y1"], locs[i]["y2"]]
            blobs.append((loc, words[i], font[i]))
    return blobs

def gather_all_blobs():
    all_blobs = []
    for url in pipeline.saved_blobs:
        for blob in pipeline.saved_blobs[url]:
            all_blobs.append((blob["coords"], blob["words"], blob["font"]))
    return all_blobs

def split_blobs(gathered_blobs, all_blobs):
    def intersect(x, y):
        return [a for a in y if a not in x]
    blobs, other = gathered_blobs, intersect(gathered_blobs, all_blobs)
    return blobs, other

def build_model(input_length=5):
    model = Sequential()
    model.add(Dense(50, activation = "relu", input_shape=(5,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = "tanh", input_shape=(5,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = "sigmoid", input_shape=(5,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = "tanh", input_shape=(5,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = "relu", input_shape=(5,)))
    model.add(Dense(1, activation = "sigmoid"))

    start = time.time()
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )
    print("Model Compilation Time: ", time.time() - start)
    model.summary()
    return model

def get_x_y(filename):
    c = gather_all_blobs()
    x, y = split_blobs(gather_blobs(read_blob_file(filename)), c)
    x = list(map(pipeline.blob_to_trainable, x))
    y = list(map(pipeline.blob_to_trainable, y))
    return np.array(x+y), np.array([1]*len(x) + [0]*len(y))

filename = "batch_1_1545682088_export"
model = build_model()
x, y = get_x_y(filename)

print("X.shape", x.shape)
print("Y.shape", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

results = model.fit(
    x_train, y_train,
    epochs = 50,
    batch_size = 500,
    validation_data = (x_train, y_train)
)

epoch_time = int(time.time())
model_directory = assets.DATA_PATH + "/ml/{0}".format(epoch_time)
os.mkdir(model_directory)
model.save_weights(model_directory + "/weights")

with open(model_directory + "/model_json", "w") as json:
    json.write(model.to_json())

print(np.mean(results.history["val_acc"]))
