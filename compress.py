"""Implements functions for compressing the WLA500 layout analysis dataset

usage: compress.py [-h] [--output OUTPUT_DIR] [--output_meta OUTPUT_JSON]
                   [--height HEIGHT] [--width WIDTH] [-o]
                   dataset metadata

Compress dataset

positional arguments:
  dataset               path to the dataset
  metadata              path to the metadata JSON file

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT_DIR   desired output directory
  --output_meta OUTPUT_JSON
                        desired output metadata file
  --height HEIGHT       desired image height (default: 384)
  --width WIDTH         desired image width (default: 256)
  -o, --overwrite       should duplicate files be overwritten
"""
import argparse
import json
import math
import os
import sys

from PIL import Image

DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 512
DEFAULT_OUTPUT_DIR = "WLA-500c"
DEFAULT_OUTPUT_META = "metadatac.json"


def main():
    """Main program.

    Returns:
        Zero on succesful termination, non-zero otherwise.
    """
    parser = argparse.ArgumentParser(description='Compress dataset')
    parser.add_argument('dataset_path',
                        metavar='dataset',
                        help='path to the dataset')
    parser.add_argument('metadata_path',
                        metavar='metadata',
                        help='path to the metadata JSON file')
    parser.add_argument('--output',
                        dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='desired output directory')
    parser.add_argument('--output_meta',
                        dest='output_json',
                        default=DEFAULT_OUTPUT_META,
                        help='desired output metadata file')
    parser.add_argument('--height',
                        type=int,
                        dest='height',
                        default=DEFAULT_HEIGHT,
                        help='desired image height (default: 384)')
    parser.add_argument('--width',
                        type=int,
                        dest='width',
                        default=DEFAULT_WIDTH,
                        help='desired image width (default: 256)')
    parser.add_argument('-o',
                        '--overwrite',
                        dest='overwrite',
                        action='store_true',
                        help='should duplicate files be overwritten')

    args = parser.parse_args()

    update_metadata(args.metadata_path, args.output_json, args.dataset_path,
                    args.height, args.width)
    compress_images(args.dataset_path, args.output_dir, args.height, args.width)

    sys.stdout.write("\033")
    print("Succesfully compressed dataset to %s with no errors." %
          args.output_dir)
    return 0


def compress_images(image_dir,
                    output_dir=DEFAULT_OUTPUT_DIR,
                    height=DEFAULT_HEIGHT,
                    width=DEFAULT_WIDTH):
    image_filenames = os.listdir(image_dir)
    num_images = len(image_filenames)

    for num_images_compressed, image_filename in enumerate(image_filenames):
        image_path = image_dir + "/" + image_filename
        compress_image(image_path, output_dir, height, width)

        percent = "{0:.0%}".format(num_images_compressed / num_images)
        print("Compressing images | %s | %s" % (percent, image_filename), end="\r")


def compress_image(image_path,
                   output_dir,
                   height=DEFAULT_HEIGHT,
                   width=DEFAULT_WIDTH):
    directory, basename = os.path.split(image_path)
    filename, extension = os.path.splitext(basename)
    output_path = output_dir + "/" + filename + ".png"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Warning: %s does not exist. Directory will be created." %
              output_dir)
    elif os.path.exists(output_path):
        print("Warning: %s already exists. Image will be overwritten." %
              output_path)

    image = Image.open(image_path)
    image = image.resize((width, height))
    image = image.convert("RGB")
    image.save(output_path, format="PNG", optimized=True, quality=75)


# Run this script before compressing images!
def update_metadata(metadata_path,
                    output_path,
                    image_dir,
                    height=DEFAULT_HEIGHT,
                    width=DEFAULT_WIDTH):
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    image_metadata = metadata["_via_img_metadata"]
    for image_key in image_metadata:
        image_filename = image_metadata[image_key]["filename"]
        if not image_filename in os.listdir(image_dir):
            continue

        image_path = os.path.join(image_dir, image_filename)
        original_width, original_height = Image.open(image_path).size

        x_scale = lambda x: math.floor(x * width / original_width)
        y_scale = lambda y: math.floor(y * height / original_height)

        for region in image_metadata[image_key]["regions"]:
            shape_attributes = region["shape_attributes"]
            if shape_attributes["name"] == "polygon":
                shape_attributes["all_points_x"] = [
                    x_scale(x) for x in shape_attributes["all_points_x"]
                ]
                shape_attributes["all_points_y"] = [
                    y_scale(y) for y in shape_attributes["all_points_y"]
                ]
            elif shape_attributes["name"] == "rect":
                shape_attributes["x"] = x_scale(shape_attributes["x"])
                shape_attributes["y"] = y_scale(shape_attributes["y"])
                shape_attributes["width"] = x_scale(shape_attributes["width"])
                shape_attributes["height"] = y_scale(shape_attributes["height"])

    with open(output_path, "w") as output_file:
        json.dump(metadata, output_file)


if __name__ == "__main__":
    main()
