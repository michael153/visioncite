"""asdf"""
import os
from xml.dom import minidom

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class PRImADataset(Dataset):
    """Abstraction for the PRImA Layout Analysis Dataset."""

    CLASSES = ('generic', 'caption', 'page-number', 'credit', 'paragraph',
               'footer', 'logo', 'heading', 'drop-capital', 'floating',
               'header', 'punch-hole')

    def __init__(self, image_dir, label_dir, transform=None):
        """Initializes a PRImADataset object.

        I'm making a few assumptions here:
        * Every file in the image_dir is an image
        * Every image has exactly one corresponding label
        * The label corresponding to an image named "00000xyz" is named either
          "00000xyz" or "pc-00000xyz". xyz represent a sequence of three digits
          here.
        * Labels use the ".xml" extension.

        Arguments:
            image_dir: The path to the dataset images (e.g, ./images-384x256).
            label_dir: The path to the dataset labels (e.g, ./labels-384x256).
            transform: Does nothing for now. I might implement this later.
        """
        self.image_filenames = [image_filename for image_filename in os.listdir(image_dir)]
        self.label_filenames = []
        for image_filename in self.image_filenames:
            # Replace the image filename extension with ".xml"
            label_filename = os.path.splitext(image_filename)[0] + ".xml"
            label_path = os.path.join(label_dir, label_filename)
            if os.path.isfile(label_path):
                self.label_filenames.append(label_filename)
            else:
                self.label_filenames.append("pc-" + label_filename)

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

    def __getitem__(self, index):

        def import_image(image_path):
            image = Image.open(image_path)
            image_array = np.asarray(image)
            image_tensor = torch.tensor(image_array)
            image_tensor = image_tensor.type(torch.FloatTensor)
            # We have to permute so that channels are located in the first axis.
            # i.e, we're going from H x W x C to C x H x W, as required by Conv2D.
            return image_tensor.permute(2, 0, 1)

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

            mask_tensor = torch.tensor(mask)
            mask_tensor = mask_tensor.type(torch.LongTensor)
            return mask_tensor

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
