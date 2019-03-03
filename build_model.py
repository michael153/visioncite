import os
import sys
import time
import json

import numpy as np
from PIL import Image, ImageOps

CONST_IMG_HEIGHT = 1800
CONST_IMG_WIDTH  = 1200

def img_to_np(img):
    arr = np.asarray(padded)
    print(arr.shape)
    return arr

def pad_img(img, height=CONST_IMG_HEIGHT, width=CONST_IMG_WIDTH):
    size = img.size
    dW = width - size[0]
    dH = height - size[1]
    padding = (dW//2, dH//2, dW - dW//2, dH - dH//2)
    padded = ImageOps.expand(img, padding, fill="white")
    padded.show()
    return padded

def scale_img(img):
    scaled = img.resize((CONST_IMG_HEIGHT, CONST_IMG_WIDTH))
    scaled.show()
    return scaled


def get_data(filename):
    return json.load(open(os.path.join(assets.DATA_PATH, 'training/{0}.train'.format(filename)), "r"))

"""
@param  dim         the dimensions of the images being trained on
        data_source contains all the data to be trained on
        num_class   the number of possible labels for each pixel
                    (default = 4: title, author(s), date, publisher info)
"""
def build_model(dim, data_source, num_class=4):
    print("Image shape: {0}".format(dim))
    flattened_dim = np.prod(np.array(dim))
    
    """
    @param  d           integer equal to m x n, where m and n are the
                        dimensions of the images being passed into the
                        model
    @param  num_class   number of possible labels for each pixel
    @return model       given an image, this model returns a d*num_class
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
    SegNet-Like Conv Net Model that takes 3D image input (RGB)
    @param  image_dim   a tuple (h, w, 3) or (h, w) that represents the
                        dimensions of the images being passed in
    @param  num_class   number of possible labels for each pixel
    @return model       given an image, model returns num_class
                        probabilities for each pixel
    """
    def make_convnet_model(image_dim, num_class):
        height, width = image_dim[0], image_dim[1]

        model = Sequential()
        model.add(Conv3D(128, kernel_size=2, activation='relu', input_shape=image_dim))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding='valid'))
        model.add(Conv3D(64, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding='valid'))
        model.add(Conv3D(32, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1), padding='valid'))
        
        model.add(UpSampling3D(size=(2,2,2)))
        model.add(Conv3D(32, kernel_size=3, activation='relu'))
        model.add(UpSampling3D(size=(2,2,2)))
        model.add(Conv3D(64, kernel_size=3, activation='relu'))
        model.add(UpSampling3D(size=(2,2,2)))
        model.add(Conv3D(128, kernel_size=3, activation='relu'))

        model.add(Flatten())
        flattened_dim = height*width*num_class
        model.add(Dense(flattened_dim, activation='softmax'))

        start = time.time()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
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

    # model = make_simple_model(arr.len, num_class)
    model = make_convnet_model(dim, num_class)
    train(model, get_data(data_source))

    return model;