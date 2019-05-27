from xml.dom import minidom
import json
from PIL import Image, ImageDraw
import numpy as np

from datasets import PRImADataset


def xml_to_json(xml_data):
    label_dom = minidom.parse(xml_data)
    data = {}

    page_element = label_dom.getElementsByTagName('Page')[0]
    metadata = {}
    metadata['filename'] = page_element.getAttribute('imageFilename')
    metadata['height'] = int(page_element.getAttribute('imageHeight'))
    metadata['width'] = int(page_element.getAttribute('imageWidth'))
    data['metadata'] = metadata

    labels = {}

    region_tags = ['TextRegion', 'ImageRegion', 'GraphicRegion']
    for tag in region_tags:

        regions = label_dom.getElementsByTagName(tag)

        for region in regions:

            if 'type' in region.attributes:
                region_type = region.getAttribute('type')
            else:
                region_type = 'generic'

            if region_type not in labels:
                labels[region_type] = []

            coords_element = region.getElementsByTagName('Coords')[0]
            points = coords_element.getElementsByTagName('Point')
            polygon = []

            for point in points:
                x_value = int(point.attributes['x'].value)
                y_value = int(point.attributes['y'].value)
                point_value = x_value, y_value
                polygon.append(point_value)

            labels[region_type].append(polygon)

        data['labels'] = labels
        return data


def json_to_mask(json_data):
    """Generates the label mask for an image given its ground truth JSON data.

    Arguments:
        json_data: The ground truth data of an image in JSON form

    Returns:
        A 2D (numpy) array of integer labels of size (height, width)
    """
    width, height = json_data['metadata']['width'], json_data['metadata']['height']

    overlay = Image.new('L', (width, height), 255)
    for region_type in json_data["labels"]:
        color = PRImADataset.CLASSES.index(region_type)
        for polygon in json_data["labels"][region_type]:
            draw = ImageDraw.Draw(overlay)
            draw.polygon(polygon, fill=color, outline=color)

    mask = np.asarray(overlay).astype(int)
    mask.setflags(write=1)
    mask[mask == 255] = 0
    return mask

