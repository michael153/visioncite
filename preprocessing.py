import json
from xml.dom import minidom
from PIL import Image

import assets

DESIRED_IMAGE_WIDTH = 1200
DESIRED_IMAGE_HEIGHT = 1800


def import_image(image_filename):
    """Resize an image and appropriately modify its ground truth data."

    The image must be a TIF files.

    Arguments:
        image_filename: The name of an image file

    Returns:
        A two-tuple containing an Image of the resized image and the modified
        ground truth data for the image.
    """
    image = Image.open(image_filename)
    width, height = image.size

    def scaling_function(point):
        x_scale = DESIRED_IMAGE_WIDTH / width
        y_scale = DESIRED_IMAGE_HEIGHT / height
        return (point[0] * x_scale, point[1] * y_scale)

    resized_image = image.resize((DESIRED_IMAGE_WIDTH, DESIRED_IMAGE_HEIGHT))
    xml_filename = image_filename[:-len(".tif")] + ".xml"
    ground_truth_data = xml_to_json(xml_filename, scaling_function)

    return (resized_image, ground_truth_data)


"""
@param  xml_source_file     name of file containing xml data
        lambda_func         scaling function that takes in
                            a tuple (x, y)
"""
def xml_to_json(xml_source_file, lambda_func=None):
    xmldoc = minidom.parse(xml_source_file)
    region_types = ['TextRegion', 'ImageRegion', 'GraphicRegion']
    data = {}
    metadata = {}
    meta = xmldoc.getElementsByTagName('Page')[0]
    metadata['filename'] = meta.attributes['imageFilename'].value
    metadata['height'] = int(meta.attributes['imageHeight'].value)
    metadata['width'] = int(meta.attributes['imageWidth'].value)
    if lambda_func:
        metadata['width'], metadata['height'] = lambda_func(
            (metadata['width'],
            metadata['height'])
        )
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
