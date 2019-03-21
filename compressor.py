"""Implements functions for compressing the PRImA layout analysis dataset."""
import os
import json

from PIL import Image

import settings
import preprocessing


def compress_image(path, output_directory_path):
    """Compresses the image at the specified file path.

    The image will be converted to JPEG and reduce to the resolution
    specified in the settings module.

    Arguments:
        path: Path to an image
        output_directory_path: Path to a directory
    """
    assert os.path.exists(
        output_directory_path), "Could not find output directory."

    image = Image.open(path)
    resized_image = image.resize(settings.DESIRED_RESOLUTION)
    rgb_image = resized_image.convert("RGB")

    # This gets the filename from the path
    path_head = os.path.split(path)[1]
    # This replaces the old file extension with ".jpeg"
    filename = os.path.splitext(path_head)[0] + ".jpeg"

    print("Saving compressed image " + filename)
    rgb_image.save(
        output_directory_path + "/" + filename,
        format="JPEG",
        optimized=True,
        quality=75)


def resize_ground_truth(path, output_directory_path):
    """Resizes ground truth data to match its corresponding image.

    If an image is compressed, the ground truth data may no longer be
    accurate. This function rescales points in the ground truth data
    so that they match the compressed image.

    XML ground truth data will be converted to JSON.

    Arguments:
        path: Path to a ground truth data file
        output_directory_path: Path to a directory
    """
    assert os.path.exists(
        output_directory_path), "Could not find output directory"

    data = preprocessing.xml_to_json(path)

    if not "metadata" in data:
        return

    def scaling_function(point):
        x_scale = settings.DESIRED_IMAGE_WIDTH / data['metadata']['height']
        y_scale = settings.DESIRED_IMAGE_HEIGHT / data['metadata']['width']
        return (int(point[0] * x_scale), int(point[1] * y_scale))

    scaled_data = preprocessing.xml_to_json(path, scaling_function)
    # This gets the filename from the path
    path_head = os.path.split(path)[1]
    # This replaces the ".xml" with ".json"
    filename = os.path.splitext(path_head)[0] + ".json"

    print("Saving scaled ground truth " + filename)
    with open(output_directory_path + "/" + filename, 'w') as file:
        json.dump(scaled_data, file)


def make_output_directories(compressed_dataset_path):
    """Creates output directory for the compressed dataset.

    Subdirectories named "json" and "images" will also be created.

    Arguments:
        output_directory_path: A path to the desired output directory location
    """
    if not os.path.exists(compressed_dataset_path):
        os.makedirs(compressed_dataset_path)
        print("Creating directory " + compressed_dataset_path)
    if not os.path.exists(compressed_dataset_path + "/images"):
        os.makedirs(compressed_dataset_path + "/images")
        print("Creating directory " + compressed_dataset_path + "images/")
    if not os.path.exists(compressed_dataset_path + "/json"):
        os.makedirs(compressed_dataset_path + "/json")
        print("Creating directory " + compressed_dataset_path + "/json")


def compress_dataset(dataset_path, compressed_dataset_path="compressed"):
    """Compresses the PRImA layout analysis dataset to a specified path.

    Arguments:
        dataset_path: Path to the dataset. Two directories, "XML" and "Images"
            must be contained inside.
        compressed_dataset_path: The desired path to the compressed dataset. The
            directory will be created if it does not currently exist.
    """
    make_output_directories(compressed_dataset_path)
    for filename in os.listdir(dataset_path + "/images"):
        compress_image(dataset_path + "/images/" + filename,
                       compressed_dataset_path + "/images")
    for filename in os.listdir(dataset_path + "/xml"):
        resize_ground_truth(dataset_path + "/xml/" + filename,
                            compressed_dataset_path + "/json")
