import os
import sys
import time
import json

import numpy as np
from PIL import Image, ImageOps
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

import assets
import settings
from preprocessing import import_image


def get_data_batch(filename):
    with open(os.path.join(assets.TRAINING_PATH, filename), 'r') as f:
        lines = [line.strip('\n') for line in f]
    return lines

"""
@param  dim         the dimensions of the images being trained on
        num_class   the number of possible labels for each pixel
                    (default = 4: title, author(s), date, publisher info)
"""
def build_model(dim, num_class=4):
    print("Image shape: {0}".format(dim))
    flattened_dim = np.prod(np.array(dim))

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
        channels = 3 #RGB

        input_shape = (height, width, channels)

        model = Sequential()
        model.add(Conv2D(128, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        model.add(Conv2D(64, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        model.add(Conv2D(32, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Conv2D(128, kernel_size=3, activation='relu'))

        model.add(Conv2D(num_class, kernel_size=3, activation='softmax'))

        start = time.time()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model Compilation Time: ", time.time() - start)
        model.summary()
        return model

    return make_convnet_model(dim, num_class)

"""
@param  model       the model being trained
@param  data_batch  list of filenames to be training
"""
def train(model, data_batch):
    X = []
    Y = []
    for file in data_batch:
        img, mask = import_image(file)
        if mask is not None:
            X.append(img)
            Y.append(mask)
    X = np.array(X)
    Y = np.array(Y)
    x_train = X[:3*len(X)//4]
    y_train = Y[:3*len(Y)//4]
    x_test = X[3*len(X)//4:]
    y_test = Y[3*len(Y)//4:]

    print("\n")
    print("x_train.shape", x_train.shape)
    print("x_test.shape", x_test.shape)
    print("x_train[0].shape", x_train[0].shape)
    print("y_train.shape", y_train.shape)
    print("y_test.shape", y_test.shape)
    print("y_test[0].shape", y_test[0].shape)
    print("\n")

    results = model.fit(
        x_train, y_train,
        epochs = 100,
        batch_size = 64,
        validation_data = (x_test, y_test)
    )

    epoch_time = int(time.time())
    model_directory = assets.DATA_PATH + "/ml/{0}".format(epoch_time)
    os.mkdir(model_directory)
    model.save_weights(model_directory + "/weights")
    with open(model_directory + "/model_json", "w") as json:
        json.write(model.to_json())
    print(np.mean(results.history["val_acc"]))
    return model


if __name__ == "__main__":
    data_batch = get_data_batch("33020190233.train")

    print(data_batch)
    model = build_model((settings.DESIRED_IMAGE_HEIGHT, settings.DESIRED_IMAGE_WIDTH), len(settings.LABELS))
    print("\n\n")
    train(model, data_batch)
