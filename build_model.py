import os
import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

import assets
import settings
from preprocessing import import_image


def main():
    image_list_filename = sys.argv[1]
    image_list_path = os.path.join(assets.TRAINING_PATH, image_list_filename)
    image_filenames = get_image_filenames(path_to_image_filenames_list)
    image_dimensions = settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH
    labels_count = len(settings.LABELS)
    model = build_model(image_dimensions, labels_count)
    train(model, image_filenames)
    model_directory = sys.argv[2]
    save_model_to_directory(model, model_directory)


def get_image_filenames(filepath):
    with open(filepath, 'r') as file:
        image_filenames = [filename.strip('\n') for filename in file]
    return image_filenames


def build_model(image_dimensions, labels_count):
    height, width = image_dimensions[0], image_dimensions[1]
    channels = 3  #RGB
    input_shape = (height, width, channels)

    model = Sequential()
    model.add(
        Conv2D(128, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(Conv2D(64, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(Conv2D(32, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))

    model.add(Conv2D(labels_count, kernel_size=3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def train(model, image_filenames, training_validation_ratio=0.75):
    unlabeled_data = []
    labeled_data = []
    for image_filename in image_filenames:
        image, mask = import_image(image_filename)
        if mask is not None:
            unlabeled_data.append(image)
            labeled_data.append(mask)
    unlabeled_data = np.array(unlabeled_data)
    labeled_data = np.array(labeled_data)

    training_input_data, validation_input_data = partition_data(
        unlabeled_data, training_validation_ratio)
    training_target_data, validation_target_data = partition_data(
        labeled_data, training_validation_ratio)

    results = model.fit(training_input_data,
                        training_target_data,
                        epochs=100,
                        batch_size=64,
                        validation_data=(validation_input_data,
                                         validation_target_data))
    return model


def partition_data(data, ratio):
    partition_index = int(ratio * len(data))
    return data[:partition_index], data[partition_index:]


def save_model_to_directory(model, directory_path=None):
    """Saves the specified model to a directory.

    If the directory is not specified, the model will be saved to the directory
    defined in the settings module. Moreover, if the directory does not exist,
    it will be created.

    Arguments:
        model: The model to save.
        directory_path: A path to the directory where the model will be saved.
    """
    if directory_path is None:
        directory_path = assets.MODELS_PATH
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    model.save_weights(directory_path + "/weights")
    with open(directory_path + "/model_json", "w") as file:
        file.write(model.to_json())


if __name__ == "__main__":
    main()
