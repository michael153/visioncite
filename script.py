import assets
import settings
import os

import json
import random
from PIL import Image, ImageDraw
from matplotlib.pyplot import cm
import numpy as np
from keras.models import model_from_json

def scale_img(img, width=settings.DESIRED_IMAGE_WIDTH, height=settings.DESIRED_IMAGE_HEIGHT):
    scaled = img.resize((width, height))
    return scaled

def load_model(epoch_id):
    file_path = assets.ML_PATH + '/{0}'.format(epoch_id)
    with open(file_path + '/model_json', 'r') as f:
        json_data = f.read()
        saved_model = model_from_json(json_data)
        saved_model.load_weights(file_path + '/weights')
        return saved_model

def test_model(model, img):
    # Greyscale colors corresponding to each class's mask color
    colors = [255 / (i+1) for i in range(len(settings.LABELS))]
    mask = model.predict(np.array([img]))[0]
    M = settings.DESIRED_IMAGE_HEIGHT
    N = settings.DESIRED_IMAGE_WIDTH
    arr = np.empty((M, N))
    arr.fill(colors[0])
    print(arr.shape)
    print(mask.shape)
    # Iterate through the ith column of probabilities and collect average, std data
    class_avg = [np.average(mask[:,:,i]) for i in range(len(settings.LABELS))]
    class_std = [np.std(mask[:,:,i]) for i in range(len(settings.LABELS))]
    print("\n\n")
    print("Avg of class probabilities:")
    print(class_avg)
    print("Std of class probabilities:")
    print(class_std)
    for i in range(M):
        for j in range(N):
            for label in range(1, len(settings.LABELS)):
                if mask[i][j][label] > class_avg[label] + 0.5*class_std[label]:
                    arr[i][j] = colors[label]
    # with open('proba.txt', 'w') as outfile:
    #     outfile.write('# Array shape: {0}\n'.format(mask.shape))
    #     for data_slice in mask:
    #         np.savetxt(outfile, data_slice, fmt='%-7.3f')
    arr = np.array(arr)
    img = Image.fromarray(arr, 'L')
    img.show()

def get_all_classes():
    images_dir = os.path.join(assets.DATA_PATH, "dataset_384_256/images/")
    json_dir = os.path.join(assets.DATA_PATH, "dataset_384_256/json/")
    all_classes = set([])
    fails = 0
    for filename in os.listdir(images_dir):
        if ".jpeg" in filename:
            try:
                print("Reading..." + filename)
                name = filename[:filename.find(".jpeg")]
                json_data = json.load(open(os.path.join(json_dir, name + ".json"), "r"))
                xml_data = json_data["xml"]
                for region in xml_data:
                    for region_class in xml_data[region]:
                        all_classes = set(list(all_classes) + [region_class])
            except:
                fails += 1
    print("Number of fails: " + str(fails))
    return all_classes

img = scale_img(Image.open(os.path.join(assets.DATA_PATH, "dataset_384_256/images/00001050.jpeg")))
test_model(load_model(1555646431), np.asarray(img))

# print(get_all_classes())