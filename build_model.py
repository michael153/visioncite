import os
import sys
import time
import json

import torch
import torch.nn as nn
import numpy as np

image_dimensions = (x, y)

def get_data(filename):
    return json.load(open(os.path.join(assets.DATA_PATH, 'training/{0}.train'.format(filename)), "r"))

"""
Given img matrix, return mask
"""
def build_model(img):
    dim = img.shape
    arr = img.flatten()
    #title, author(s), date, publisher info
    num_class = 4;
    filename = "replace_this_text"

    """
    @param  d       m x n, where m and n are the dimensions of the images
                    being passed into the model
    @return model   given an image, this model returns a d*num_class
                    array of probabilities. (num_class*i, num_class*(i+1))
                    represents the softmax probabilities of the ith
                    pixel in respect to the classes
    """
    def make_simple_model(d, num_class):
        model = Sequential(d)
        model.add(Dense(d/2, activation = "relu", input_shape=(d,)))
        model.add(Dropout(0.1))
        model.add(Dense(d/4, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(d/8, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(d/8, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(d/4, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(d/2, activation = "relu"))
        model.add(Dropout(0.1))
        model.add(Dense(d, activation = "relu"))
        model.add(Dense(d*num_class, activation = "softmax"))
        start = time.time()
        model.compile(
            optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = ["accuracy"]
        )
        print("Model Compilation Time: ", time.time() - start)
        model.summary()
        return model

    """
    @param  training_data   array of dicts, each dict has "input" and "expected_output" keys 
    """
    def train(model, training_data):
        data_size = len(training_data)
        set1, set2 = training_data[:data_size*0.75], training_data[data_size*0.75:]
        x_train = np.array([x["input"] for x in set1])
        y_train = np.array([y["expected_output"] for y in set1])
        x_test = np.array([x["input"] for x in set2])
        y_test = np.array([y["expected_output"] for y in set2])
        print("\n")
        print("training_data.len", training_data.len)
        print("x_train.shape", x_train.shape)
        print("x_test.shape", x_test.shape)
        print("y_train.shape", y_train.shape)
        print("y_test.shape", y_test.shape)
        print("\n")
        results = model.fit(
            x_train, y_train,
            epochs = 2500,
            batch_size = 1000,
            validation_data = (x_test, y_test)
        )
        epoch_time = int(time.time())
        model_directory = assets.DATA_PATH + "/ml/{0}".format(epoch_time)
        os.mkdir(model_directory)
        model.save_weights(model_directory + "/weights")
        with open(model_directory + "/model_json", "w") as json:
            json.write(model.to_json())
        print(np.mean(results.history["val_acc"]))


    model = make_simple_model(arr.len, num_class)
    train(model, get_data(filename))

    return model;


# def create_conv(c1, c2, c3):
#     model = nn.Sequential(
#         nn.Conv2d(c1, c2, kernel_size=(3, 3)),
#         nn.BatchNorm2d(c2),
#         nn.ReLU(),
#         nn.Conv2d(c2, c3, kernel_size=(3, 3)),
#         nn.BatchNorm2d(c3),
#         nn.ReLU()
#     )
#     return model

# def create_deconv(c1, c2, c3):
#     model = nn.Sequential(
#         nn.ConvTranspose2d(c1, c2, kernel_size=(3, 3)),
#         nn.BatchNorm2d(c2),
#         nn.ReLU(),
#         nn.ConvTranspose2d(c2, c3, kernel_size=(3, 3)),
#         nn.BatchNorm2d(c3),
#         nn.ReLU()
#     )

# def create_model(num_class):
#     conv1 = create_conv(3, 64, 128)
#     conv2 = create_conv(128, 128, 128)
#     conv3 = create_conv(128, 128, 128)
#     conv4 = create_conv(128, 128, 128)
    
#     pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
#     pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
#     pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    local deconv4 = create_deconv(128, 128, 128)
    local deconv3 = create_deconv(128+128, 128, 128)
    local deconv2 = create_deconv(128+128, 128, 128)
    local deconv1 = create_deconv(128+128, 64, num_class)

    


create_conv(3, 64, 128)

