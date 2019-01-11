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
import math
import cv2
import numpy as np

import assets
import images

def get_images():
    ''' Returns a list of all images in directory'''
    files = [f.replace('.jpg', '') for f in os.listdir(images.RAW_IMAGES_PATH) if os.path.isfile(os.path.join(images.RAW_IMAGES_PATH, f))]
    return files

def get_bounded_images():
    ''' Returns a list of all the images that have bounded versions'''
    segment_data = [f.replace('.segment', '') for f in os.listdir(assets.SEGMENTS_PATH) if os.path.isfile(os.path.join(assets.SEGMENTS_PATH, f))]
    bounded_imgs = [f.replace('.jpg', '') for f in os.listdir(images.BOUNDED_IMAGES_PATH) if os.path.isfile(os.path.join(images.BOUNDED_IMAGES_PATH, f))]
    intersect = list(set(segment_data) & set(bounded_imgs)) 
    return intersect

def get_segments():
    ''' Returns a dict of the polygonal semantic segmentation data'''
    segment_files = [f for f in os.listdir(assets.SEGMENTS_PATH) if os.path.isfile(os.path.join(assets.SEGMENTS_PATH, f))]
    segment_data = {}
    for file in segment_files:
        img = file.replace('.segment', '')
        with open(os.path.join(assets.SEGMENTS_PATH, file), encoding='utf-8', errors='ignore') as f:
            get_pts = False
            pts = 0
            points = []
            polygons = []
            for line in f:
                if "[" in line and "]" in line:
                    get_pts = True
                elif get_pts:
                    pts = int(line.strip())
                    get_pts = False
                else:
                    if pts > 0:
                        points.append(eval(line.strip()))
                        pts -= 1
                    if pts == 0:
                        polygons.append(points)
                        points = []
        segment_data[img] = polygons
    return segment_data

def aggregate_labels(export_file_name):
    export_data = json.load(open(os.path.join(assets.DATA_PATH, 'web_generated/{0}'.format(export_file_name))))
    segment_data = get_segments()
    labeled_data, regular_data = [], []

    def get_rect(polygon):
        x, y = min([p[0] for p in polygon]), min([p[1] for p in polygon])
        length = max([p[0] for p in polygon]) - x
        height = max([p[1] for p in polygon]) - y
        return ((x, y), length, height)

    def get_h_v_score(file, rect):
        img = cv2.imread(os.path.join(images.RAW_IMAGES_PATH, "{0}.jpg".format(file)), 0)
        img = img[rect[0][1]:rect[0][1]+rect[2], rect[0][0]:rect[0][0]+rect[1]]
        img = 255 - img
        h_hist = np.sum(img, axis=0).tolist()
        v_hist = np.sum(img, axis=1).tolist()
        h_hist = [v/max(max(h_hist),1) for v in h_hist][::math.ceil(len(h_hist)/10)]
        v_hist = [v/max(max(v_hist),1) for v in v_hist][::math.ceil(len(v_hist)/10)]
        if len(h_hist) < 10:
            h_hist.extend([0]*(10 - len(h_hist)))
        if len(v_hist) < 10:
            v_hist.extend([0]*(10 - len(v_hist)))
        return (h_hist, v_hist)
   
    def get_neighbor_data(rect, polygons):
        x = rect[0][0]
        y = rect[0][1]
        length = rect[1]
        height = rect[2]
        total, right, left, upper, lower = 0, 0, 0, 0, 0
        if length:
            for p_id, polygon in enumerate(segments):
                (x2, y2), poly_length, poly_height = get_rect(polygon)
                intersect = lambda rect1, rect2: not (rect1[0][0]+rect1[1] < rect2[0][0] or
                    rect1[0][0] > rect2[0][0]+rect2[1] or 
                    rect1[0][1] > rect2[0][1]+rect2[2] or
                    rect1[0][1]+rect1[2] < rect2[0][1]
                )
                if not intersect(((x, y), length, height), ((x2, y2), poly_length, poly_height)):
                    total += 1
                    right_offset = (x + length) - x2
                    left_offset = (x2 + poly_length) - x
                    if x2 + poly_length > x + length and (right_offset < 0 or right_offset/length < 0.5):
                        right += 1
                    elif x2 < x and (left_offset < 0 or left_offset/length < 0.5):
                        left += 1
                    elif y2 > y + height:
                        lower += 1
                    elif y > y2 + poly_height:
                        upper += 1
        if total and length:
            return (right/total, left/total, upper/total, lower/total)
        else:
            return (0, 0, 0, 0)

    def vectorize(rect):
        (x, y), length, height = rect
        center = (x + length//2, y + height//2)
        h_hist, v_hist = get_h_v_score(image, rect)
        right, left, upper, lower = get_neighbor_data(rect, segments)
        vec = [center[0], center[1], length, height, right, left, upper, lower]
        vec.extend(h_hist)
        vec.extend(v_hist)
        return vec

    for image in export_data:
        all_segments = segment_data[image]
        rects, segments = [], []
        for t_id in range(len(export_data[image]["locs"])):
            x, rx = export_data[image]["locs"][t_id]["x1"], export_data[image]["locs"][t_id]["x2"]
            y, ry = export_data[image]["locs"][t_id]["y1"], export_data[image]["locs"][t_id]["y2"]
            length = rx - x
            height = ry - y
            rects.append(((x, y), length, height))
        for segment in all_segments:
            if get_rect(segment) not in rects:
                segments.append(segment)
        for rect in rects:
            vec = vectorize(rect)
            labeled_data.append(vec)
        for segment in segments:
            vec = vectorize(get_rect(segment))
            regular_data.append(vec)

    return {"labeled": labeled_data, "other": regular_data}

def get_citation_data():
    data_dict = json.load(open(assets.DATA_PATH + "/data_dict.json", "r"))
    return data_dict

def get_image_mappings_json():
    '''Get image IDs'''
    image_dict = json.load(open(assets.DATA_PATH + "/image_mappings.json", "r"))
    return image_dict

# def get_saved_blobs_json():
#     ''' Returns a json containing all the current blob information'''
#     saved_blobs = json.load(open(assets.DATA_PATH + "/saved_blobs.json", "r"))
#     for key in saved_blobs.copy():
#         if not os.path.isfile(images.BOUNDED_IMAGES_PATH + '/bounded_{0}.jpg'.format(image_dict[key])):
#             print("{0} not found...".format(images.BOUNDED_IMAGES_PATH + '/bounded_{0}.jpg'.format(image_dict[key])))
#             saved_blobs.pop(key, None)
#     return saved_blobs

# def save_blobs(url, blobs):
#     saved_blobs[url] = []
#     for blob in blobs:
#         blob_info = {
#             'coords': list(blob[0]),
#             'words': blob[1],
#             'font': blob[2]
#         }
#         saved_blobs[url].append(blob_info)
#     with open(assets.DATA_PATH + '/saved_blobs.json', 'w') as out:
#         json.dump(saved_blobs, out, sort_keys=True, indent=4)

def save_image_mappings(image_dict):
    with open(assets.DATA_PATH + '/image_mappings.json', 'w') as out:
        json.dump(image_dict, out, sort_keys=True, indent=4)

# def blob_to_trainable(blob):
#     '''Convert a blob object to a vectorized list that is
#     trainable
#     '''
#     return list(blob[0]) + [blob[2]]

image_dict = get_image_mappings_json()
# saved_blobs = get_saved_blobs_json()
data_dict = get_citation_data()