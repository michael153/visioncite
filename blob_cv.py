# Copyright 2018 Michael Wan
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
import json
import math
import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial as spatial
import scipy.signal as signal
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
from PIL import Image, ImageDraw

import images
import assets
import pipeline

class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        '''
        (x1, y1) is the upper left corner,
        (x2, y2) is the lower right corner,
        with (0, 0) being in the upper left corner.
        '''
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def taxicab_diagonal(self):
        '''
        Return the taxicab distance from (x1,y1) to (x2,y2)
        '''
        return self.x2 - self.x1 + self.y2 - self.y1
    def overlaps(self, other):
        '''
        Return True iff self and other overlap.
        '''
        return not ((self.x1 > other.x2)
                    or (self.x2 < other.x1)
                    or (self.y1 > other.y2)
                    or (self.y2 < other.y1))

    def __hash__(self):
        return hash(tuple((self.x1, self.x2, self.y1, self.y2)))

    def __eq__(self, other):
        return (self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2)

def find_paws(data, smooth_radius = 5, threshold = 0.0001):
    # https://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection
    """Detects and isolates contiguous regions in the input array"""
    # Blur the input data a bit so the paws have a continous footprint 
    data = ndimage.uniform_filter(data, smooth_radius)
    # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
    thresh = data > threshold
    # Fill any interior holes in the paws to get cleaner regions...
    filled = ndimage.morphology.binary_fill_holes(thresh)
    # Label each contiguous paw
    coded_paws, num_paws = ndimage.label(filled)
    # Isolate the extent of each paw
    # find_objects returns a list of 2-tuples: (slice(...), slice(...))
    # which represents a rectangular box around the object
    data_slices = ndimage.find_objects(coded_paws)
    return data_slices

def slice_to_bbox(slices):
    for s in slices:
        dy, dx = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

def remove_overlaps(bboxes):
    '''
    Return a set of BBoxes which contain the given BBoxes.
    When two BBoxes overlap, replace both with the minimal BBox that contains both.
    '''
    # list upper left and lower right corners of the Bboxes
    corners = []

    # list upper left corners of the Bboxes
    ulcorners = []

    # dict mapping corners to Bboxes.
    bbox_map = {}

    for bbox in bboxes:
        ul = (bbox.x1, bbox.y1)
        lr = (bbox.x2, bbox.y2)
        bbox_map[ul] = bbox
        bbox_map[lr] = bbox
        ulcorners.append(ul)
        corners.append(ul)
        corners.append(lr)        

    # Use a KDTree so we can find corners that are nearby efficiently.
    tree = spatial.KDTree(corners)
    new_corners = []
    for corner in ulcorners:
        bbox = bbox_map[corner]
        # Find all points which are within a taxicab distance of corner
        indices = tree.query_ball_point(
            corner, bbox_map[corner].taxicab_diagonal(), p = 1)
        for near_corner in tree.data[indices]:
            near_bbox = bbox_map[tuple(near_corner)]
            if bbox != near_bbox and bbox.overlaps(near_bbox):
                # Expand both bboxes.
                # Since we mutate the bbox, all references to this bbox in
                # bbox_map are updated simultaneously.
                bbox.x1 = near_bbox.x1 = min(bbox.x1, near_bbox.x1)
                bbox.y1 = near_bbox.y1 = min(bbox.y1, near_bbox.y1) 
                bbox.x2 = near_bbox.x2 = max(bbox.x2, near_bbox.x2)
                bbox.y2 = near_bbox.y2 = max(bbox.y2, near_bbox.y2) 
    return set(bbox_map.values())

def estimate_greyscale_noise(I):
    # https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
    H, W = I.shape
    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * ((0.5 * math.pi)**0.5) / (6 * (W-2) * (H-2))
    return sigma

def remove_bg(im):
    color = max(im.getcolors(im.size[0]*im.size[1]))[1]
    # print("Determined Bg Color (greyscale): {0}".format(color))
    if abs(color - 255) <= 10:
        return np.asarray(im)
    data = np.asarray(im)
    data.flags.writeable = True
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if abs(data[i][j] - color) <= 5:
                data[i][j] = 255
            else:
                data[i][j] = max(0, min(255 - abs(color - data[i][j]), 255))
    return data

def bounding_box(file_id, display=False, saveImage=False):
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    data = Image.open(images.RAW_IMAGES_PATH + '/{0}.jpg'.format(file_id))
    if saveImage:
        bounded_im = data.copy()
    data = data.convert('L')

    # Grey Dilation
    data = np.asarray(data)
    # Background Removal
    data = remove_bg(Image.fromarray(data))
    data = 255-ndimage.grey_dilation(255-data, size=(1,1))
    
    # Convert data back to Image (from np array)
    # https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array/37675266
    if display:
        im = ax.imshow(Image.fromarray(data))
    
    # Calculate noise of image
    score = estimate_greyscale_noise(data)
    # Cap score from [1, 10]
    score = min(max(score, 1), 10)
    # Calculate smooth parameter given by the function 64.783e^(-0.385x). Obsolete: 30e^(1.5-x)
    offset, exp = 0.2, 1.5
    smooth_function = lambda score: round((60/math.log(1.3*score))*math.e**(-0.385*score), 2) # 64.783
    smooth_parameter = min(max(smooth_function(score), 1), 60)
    # print("Image {0} has noise score: {1}".format(file_id, score))
    # print("Using smooth parameter: {0}\n\n\n".format(smooth_parameter))
    data_slices = find_paws(255-data, smooth_radius=smooth_parameter, threshold=18.5)
    # data_slices = find_paws(255-data, smooth_radius=40, threshold=18.5)

    bboxes = remove_overlaps(slice_to_bbox(data_slices))
    blobs = []
    for bbox in bboxes:
        xwidth = bbox.x2 - bbox.x1
        ywidth = bbox.y2 - bbox.y1
        p = patches.Rectangle((bbox.x1, bbox.y1), xwidth, ywidth,
                              fc = 'none', ec = 'red')
        
        bounded_box_im = data[bbox.y1:bbox.y2,bbox.x1:bbox.x2]
        if display:
            ax.add_patch(p)

        bounded_words = pytesseract.image_to_string(bounded_box_im)
        if len(bounded_words) > 0:
            # print(bbox.x1, bbox.x2, bbox.y1, bbox.y2, ": ")
            # print(bounded_words)
            font_size = abs(bbox.y2 - bbox.y1)*abs(bbox.x2 - bbox.x1)/len(bounded_words)
            # print("Rough Relative Height Calculation: {0}".format(font_size))
            # print("\n")
            blobs.append(((bbox.x1, bbox.x2, bbox.y1, bbox.y2), bounded_words, font_size))
            if saveImage:
                draw = ImageDraw.Draw(bounded_im)
                draw.rectangle(((bbox.x1, bbox.y1), (bbox.x2, bbox.y2)), outline='red')
    if display:
        plt.show()
    if saveImage:
        bounded_im.save(images.BOUNDED_IMAGES_PATH + '/bounded_{0}.jpg'.format(file_id), "JPEG")
    return blobs

def get_blobs_by_url(url, image_id, display=False, saveImage=False, useSavedData=False):
    if url in pipeline.saved_blobs and useSavedData:
        try:
            print("Extracting saved blob for URL={0}...".format(url))
            blobs_saved = pipeline.saved_blobs[url]
            blobs = []
            for blob in blobs_saved:
                x1, x2, y1, y2 = blob["coords"][0], blob["coords"][1], blob["coords"][2], blob["coords"][3]
                blobs.append(((x1, x2, y1, y2), blob["words"], blob["font"]))
            return (url, blobs)
        except Exception as e:
            print("*** Error extracting saved blob: {0}".format(e))
            raise Exception("Error extracting saved blob")
    else:
        try:
            print("Saving new blob for URL={0}...".format(url))
            blobs = bounding_box(image_id, display, saveImage)
            pipeline.save_blobs(url, blobs)
            return (url, blobs)
        except Exception as e:
            print("*** Error saving new blob: {0}".format(e))
            raise Exception("Error saving new blob")

if __name__ == '__main__':
    for k, v in pipeline.image_dict.items():
        try:
            get_blobs_by_url(k, pipeline.image_dict[k], saveImage=True, useSavedData=True)
        except:
            continue