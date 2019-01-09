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

import assets
import images

def get_bounded_images():
    ''' Returns a list of all the images that have bounded versions'''
    bounded_dict = json.load(open(assets.DATA_PATH + "/saved_blobs.json", "r"))
    return list(bounded_dict.keys())

# def get_image

def get_citation_data():
    data_dict = json.load(open(assets.DATA_PATH + "/data_dict.json", "r"))
    return data_dict

def get_image_mappings_json():
    '''Get image IDs'''
    image_dict = json.load(open(assets.DATA_PATH + "/image_mappings.json", "r"))
    return image_dict

def get_saved_blobs_json():
    ''' Returns a json containing all the current blob information'''
    saved_blobs = json.load(open(assets.DATA_PATH + "/saved_blobs.json", "r"))
    for key in saved_blobs.copy():
        if not os.path.isfile(images.BOUNDED_IMAGES_PATH + '/bounded_{0}.jpg'.format(image_dict[key])):
            print("{0} not found...".format(images.BOUNDED_IMAGES_PATH + '/bounded_{0}.jpg'.format(image_dict[key])))
            saved_blobs.pop(key, None)
    return saved_blobs

def save_blobs(url, blobs):
    saved_blobs[url] = []
    for blob in blobs:
        blob_info = {
            'coords': list(blob[0]),
            'words': blob[1],
            'font': blob[2]
        }
        saved_blobs[url].append(blob_info)
    with open(assets.DATA_PATH + '/saved_blobs.json', 'w') as out:
        json.dump(saved_blobs, out, sort_keys=True, indent=4)

def save_image_mappings(image_dict):
    with open(assets.DATA_PATH + '/image_mappings.json', 'w') as out:
        json.dump(image_dict, out, sort_keys=True, indent=4)

def blob_to_trainable(blob):
    '''Convert a blob object to a vectorized list that is
    trainable
    '''
    return list(blob[0]) + [blob[2]]

image_dict = get_image_mappings_json()
saved_blobs = get_saved_blobs_json()
data_dict = get_citation_data()