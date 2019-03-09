import os
import json

from PIL import Image

import settings
import preprocessing

OUTPUT_DIRECTORY_PATH = "compressed/"
IMAGE_SUBDIRECTORY_PATH = OUTPUT_DIRECTORY_PATH + "images/"
XML_SUBDIRECTORY_PATH = OUTPUT_DIRECTORY_PATH + "xml/"


def compress_image(path):
    image = Image.open(path)
    resized_image = image.resize(settings.DESIRED_RESOLUTION)
    rgb_image = resized_image.convert("RGB")

    path_head = os.path.split(path)[1]
    filename = os.path.splitext(path_head)[0] + ".jpeg"

    print("Saving compressed image " + filename)
    rgb_image.save(IMAGE_SUBDIRECTORY_PATH + filename, format="JPEG", optimized=True, quality=75)


def resize_ground_truth(path):
    data = preprocessing.xml_to_json(path)
    if not "metadata" in data:
        return

    def scaling_function(point):
        x_scale = settings.DESIRED_IMAGE_WIDTH / data['metadata']['height']
        y_scale = settings.DESIRED_IMAGE_HEIGHT / data['metadata']['width']
        return (int(point[0] * x_scale), int(point[1] * y_scale))

    scaled_data = preprocessing.xml_to_json(path, scaling_function)
    path_head = os.path.split(path)[1]
    filename = os.path.splitext(path_head)[0] + ".json"
    print(filename)

    print("Saving scaled ground truth " + filename)
    with open(XML_SUBDIRECTORY_PATH + filename, 'w') as file:
        json.dump(scaled_data, file)


def make_directories():
    if not os.path.exists(OUTPUT_DIRECTORY_PATH):
        os.makedirs(OUTPUT_DIRECTORY_PATH)
        print("Creating directory " + OUTPUT_DIRECTORY_PATH)
    if not os.path.exists(IMAGE_SUBDIRECTORY_PATH):
        os.makedirs(IMAGE_SUBDIRECTORY_PATH)
        print("Creating directory " + IMAGE_SUBDIRECTORY_PATH)
    if not os.path.exists(XML_SUBDIRECTORY_PATH):
        os.makedirs(XML_SUBDIRECTORY_PATH)
        print("Creating directory " + XML_SUBDIRECTORY_PATH)


def compress_dataset(path):
    make_directories()
    for filename in os.listdir(path + "/images"):
        # compress_image(path + "/images/" + filename)
        pass
    for filename in os.listdir(path + "/xml"):
        resize_ground_truth(path + "/xml/" + filename)
