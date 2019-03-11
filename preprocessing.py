import os
import json
import numpy as np
from xml.dom import minidom
from PIL import Image, ImageDraw
from matplotlib.path import Path

import assets
import settings

def import_image(image_filename):
    """Resize an image and appropriately modify its ground truth data."

    The image must be a TIF files.

    Arguments:
        image_filename: The name of an image file

    Returns:
        A two-tuple containing an Image of the resized image and the modified
        ground truth data for the image.
    """
    image = Image.open(os.path.join(assets.IMAGE_PATH, image_filename))
    width, height = image.size

    def scaling_function(point):
        x_scale = settings.DESIRED_IMAGE_WIDTH / width
        y_scale = settings.DESIRED_IMAGE_HEIGHT / height
        return (int(point[0] * x_scale), int(point[1] * y_scale))

    resized_image = image.resize((settings.DESIRED_IMAGE_WIDTH, settings.DESIRED_IMAGE_HEIGHT))
    xml_filename = image_filename[:-len(".tif")] + ".xml"
    ground_truth_data = xml_to_json(os.path.join(assets.XML_PATH, xml_filename), scaling_function)
    
    pixels = np.copy(np.asarray(resized_image))
    mask = json_to_mask(ground_truth_data)

    return (pixels, mask)


def xml_to_json(xml_source_file, lambda_func=None):
    """Parses the XML ground truth data downloaded from the Parsa dataset
    into more easily manipulable.

    The keywords must be in the correct format and capitalization. Need
    to make sure that the XML formatting throughout the dataset is
    consistent (I know for a fact that 00000122.xml is formatted
    differently).

    Arguments:
        xml_source_file: The name of the file containing XML data
        lambda_func: An optional scaling function parameter that takes
                     in a tuple (x, y) and scales it by a defined amount
    """
    try:
        xmldoc = minidom.parse(xml_source_file)
        region_types = ['TextRegion', 'ImageRegion', 'GraphicRegion']
        data = {}
        metadata = {}
        meta = xmldoc.getElementsByTagName('Page')[0]
        metadata['filename'] = meta.attributes['imageFilename'].value
        metadata['height'] = int(meta.attributes['imageHeight'].value)
        metadata['width'] = int(meta.attributes['imageWidth'].value)
        if lambda_func:
            metadata['width'], metadata['height'] = (settings.DESIRED_IMAGE_WIDTH,
                settings.DESIRED_IMAGE_HEIGHT)
        json = {}
        for t in region_types:
            regions = xmldoc.getElementsByTagName(t)
            json[t] = {}
            json[t]['generic'] = []
            for region in regions:
                wrapper = region.getElementsByTagName('Coords')[0]
                points = wrapper.getElementsByTagName('Point')
                coords = []
                for point in points:
                    p = (int(point.attributes['x'].value), int(point.attributes['y'].value))
                    if lambda_func:
                        p = lambda_func(p)
                    coords.append(p)
                if 'type' in region.attributes:
                    if region.attributes['type'].value not in json[t]:
                        json[t][region.attributes['type'].value] = []
                    json[t][region.attributes['type'].value].append(coords)
                else:
                    json[t]['generic'].append(coords)
        data['metadata'] = metadata
        data['xml'] = json
        return data
    except:
        print("Error reading XML file: {0}".format(xml_source_file))
        return {}


def json_to_mask(json_data):
    """Generates the label mask for an image given its ground truth JSON data.
    
    Arguments: 
        json_data: The ground truth data of an image in JSON form

    Returns:
        A 2D (numpy) array of integer labels of size (height, width)
    """
    width, height = (json_data['metadata']['width'], json_data['metadata']['height'])
    overlay = Image.new('L', (width, height), 255)
    data = json_data['xml']
    for region in data:
        for region_type in data[region]:
            if region_type == 'generic':
                color = settings.LABELS.index(region_type + "_" + region)
            else:
                color = settings.LABELS.index(region_type)
            for poly in data[region][region_type]:
                poly = [tuple(p) for p in poly]
                draw = ImageDraw.Draw(overlay)
                draw.polygon(poly, fill=color, outline=color)
    mask = np.copy(np.asarray(overlay)).astype(int)
    mask.setflags(write = 1)
    mask[mask == 255] = 0
    one_hots = np.zeros((mask.shape[0], mask.shape[1], len(settings.LABELS)))
    for i in range(one_hots.shape[0]):
        for j in range(one_hots.shape[1]):
            one_hots[i][j][mask[i][j]] = 1
    return one_hots