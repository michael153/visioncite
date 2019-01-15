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

def get_data(filename):
    return json.load(open(os.path.join(assets.DATA_PATH, 'training/{0}.train'.format(filename)), "r"))

def get_x_y(labeled, regular):
    x = np.array(labeled + regular)
    y = np.array([1]*len(labeled) + [0]*len(regular))
    return x, y

'''
Vector shape:
[cx, cy, len, height, right, left, upper, lower, h_array(10), v_array(10)]
(28, 1)
'''
def build_model(input_length=28):
    model = Sequential()
    model.add(Dense(250, activation = "relu", input_shape=(input_length,)))
    model.add(Dense(250, activation = "tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation = "sigmoid"))
    model.add(Dense(100, activation = "tanh"))
    model.add(Dropout(0.3))
    model.add(Dense(40, activation = "sigmoid"))
    model.add(Dense(40, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation = "sigmoid"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dropout(0.2))
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

# file_name = "1547083109_export"
# file_name = '1547083109_export_v1547432879'
file_name = 'bad_polys_v1547455536'
data = get_data(file_name)
X, Y = get_x_y(data["labeled"], data["other"])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

print("\n")
print("X.shape", X.shape)
print("Y.shape", Y.shape)
print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)
print("\n")

model = build_model(8)

results = model.fit(
    x_train, y_train,
    epochs = 100,
    batch_size = 750,
    validation_data = (x_train, y_train)
)

epoch_time = int(time.time())
model_directory = assets.DATA_PATH + "/ml/{0}".format(epoch_time)
os.mkdir(model_directory)
model.save_weights(model_directory + "/weights")

with open(model_directory + "/model_json", "w") as json:
    json.write(model.to_json())

print(np.mean(results.history["val_acc"]))
