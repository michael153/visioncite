"""Implements custom datasets."""
import json
import os
from xml.dom import minidom

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def import_image(image_path):
    image = Image.open(image_path)
    image_array = np.asarray(image)
    image_tensor = torch.tensor(image_array, dtype=torch.float)
    # Remove alpha channel, if one exists
    image_tensor = image_tensor[:, :, :3]
    # We have to permute so that channels are located in the first axis.
    # i.e, we're going from H x W x C to C x H x W, as required by Conv2D.
    return image_tensor.permute(2, 0, 1)


class VIADataset(Dataset):

    CLASSES = ("title", "author", "date", "website", "image", "paragraph",
               "publisher", "other")

    def __init__(self, image_dir, label_json, transform=None):
        self.transform = transform

        self.image_filenames = []
        for image_filename in os.listdir(image_dir):
            self.image_filenames.append(image_filename)

        self.filename_to_image = {}
        for image_filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_filename)
            image = import_image(image_path)
            self.filename_to_image[image_filename] = image

        def draw_poly(shape_attributes, image, color):
            draw = ImageDraw.Draw(image)
            x_values = shape_attributes["all_points_x"]
            y_values = shape_attributes["all_points_y"]
            xy_points = list(zip(x_values, y_values))
            draw.polygon(xy_points, fill=color, outline=color)

        #pylint: disable=invalid-name
        def draw_rect(shape_attributes, image, color):
            draw = ImageDraw.Draw(image)
            x0, y0 = shape_attributes["x"], shape_attributes["y"]
            x1 = x0 + shape_attributes["width"]
            y1 = y0 + shape_attributes["height"]
            xy_points = [(x0, y0), (x1, y1)]
            draw.rectangle(xy_points, fill=color, outline=color)

        def create_mask(regions, size):
            overlay = Image.new('L', size, len(self.CLASSES) - 1)

            for region in regions:
                shape_attributes = region["shape_attributes"]
                shape_type = shape_attributes["name"]

                region_attributes = region["region_attributes"]
                if type(region_attributes["type"]) is str:
                    # Balaji's data is set up this way
                    region_label = region["region_attributes"]["type"]
                else:
                    # Michael's data is set up this way
                    for key in region_attributes["type"]:
                        if region_attributes["type"][key]:
                            region_label = key

                if not region_label in self.CLASSES:
                    continue
                color = self.CLASSES.index(region_label)

                if shape_type == "rect":
                    draw_rect(shape_attributes, overlay, color)
                elif shape_type == "polygon":
                    draw_poly(shape_attributes, overlay, color)

            mask = np.asarray(overlay).astype(int)
            return torch.tensor(mask, dtype=torch.long)

        with open(label_json) as json_file:
            image_metadata = json.load(json_file)["_via_img_metadata"]

        self.filename_to_mask = {}
        for image_key in image_metadata:
            filename = image_metadata[image_key]["filename"]
            if not filename in self.image_filenames:
                continue

            image = self.filename_to_image[filename]
            height, width = image.shape[1], image.shape[2]

            regions = image_metadata[image_key]["regions"]
            mask = create_mask(regions, (width, height))
            self.filename_to_mask[filename] = mask



    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image = self.filename_to_image[image_filename]
        mask = self.filename_to_mask[image_filename]

        if self.transform:
            image = self.transform(image)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)


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
        self.image_filenames = [
            image_filename for image_filename in os.listdir(image_dir)
        ]
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
