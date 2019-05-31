"""asdf"""
import os
from xml.dom import minidom

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class PRImADataset(Dataset):
    """Abstraction for the PRImA Layout Analysis Dataset."""

    CLASSES = (
        'generic',
        'caption',
        'page-number',
        'credit',
        'paragraph',
        'footer',
        'logo',
        'heading',
        'drop-capital',
        'floating',
        'header',
        'punch-hole'
    )

    def __init__(self, data_file, image_dir, label_dir, transform=None):
        """Initializes a PRImADataset object.

        The image directory and label directory should correspond to data of
        matching dimensions.

        The train file should contain newline-separated pairs of image-label
        filenames. For example, "00000086.jpeg 00000086.xml".

        Arguments:
            data_file: A path to a train file (e.g, ./32020191045.train).
            image_dir: The path to the dataset images (e.g, ./images-384x256).
            label_dir: The path to the dataset labels (e.g, ./labels-384x256).
            transform: Does nothing for now. I might implement this later.
        """
        with open(data_file) as file:
            self.image_filenames = []
            self.label_filenames = []
            for line in file.readlines():
                image_filename, label_filename = line.split()
                self.image_filenames.append(image_filename)
                self.label_filenames.append(label_filename)

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

    def __getitem__(self, index):

        def import_image(image_path):
            image = Image.open(image_path)
            image_array = np.asarray(image)
            return image_array

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
            width = json_data['metadata']['width']
            height = json_data['metadata']['height']

            overlay = Image.new('L', (width, height), 255)
            for region_type in json_data["labels"]:
                color = self.CLASSES.index(region_type)
                for polygon in json_data["labels"][region_type]:
                    draw = ImageDraw.Draw(overlay)
                    draw.polygon(polygon, fill=color, outline=color)

            mask = np.asarray(overlay).astype(int)

            # Set unlabeled areas to zero
            mask.setflags(write=1)
            mask[mask == 255] = 0

            return mask

        # Load image
        image_filename = self.image_filenames[index]
        image_path = self.image_dir + "/" + image_filename
        image = import_image(image_path)

        # Load label corresponding to above image
        label_filename = self.label_filenames[index]
        label_path = os.path.join(self.label_dir, label_filename)
        label_json = xml_to_json(label_path)
        mask = json_to_mask(label_json)

        sample = {"image": image, "label": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_filenames)


def visualize_sample(sample):
    image, mask = sample["image"], sample["label"]
    plt.imshow(image)
    plt.imshow(mask, alpha = 0.5)
