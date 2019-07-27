"""Implements functions for compressing the PRImA layout analysis dataset.

usage: compress.py [-h] [--output OUTPUT_DIR] [--height HEIGHT]
                   [--width WIDTH] [-o]
                   dataset

Compress dataset

positional arguments:
  dataset              path to the dataset

optional arguments:
  -h, --help           show this help message and exit
  --output OUTPUT_DIR  desired output directory
  --height HEIGHT      desired image height (default: 384)
  --width WIDTH        desired image width (default: 256)
  -o, --overwrite      should duplicate files be overwritten
"""
# TODO: Add support for arbitrary image extensions
import argparse
import os
import sys

from PIL import Image

DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 512
DEFAULT_OUTPUT_DIR = "dataset-%s-%s" % (DEFAULT_HEIGHT, DEFAULT_WIDTH)
# If this is set to true, then the compressor will overwrite duplicate images.
# Otherwise, duplicate images will be skipped.
DEFAULT_OVERWRITE = True

IMAGE_EXTENSION = ".png"


def main():
    """Main program.

    Returns:
        Zero on succesful termination, non-zero otherwise.
    """
    parser = argparse.ArgumentParser(description='Compress dataset')
    parser.add_argument('dataset_path',
                        metavar='dataset',
                        help='path to the dataset')
    parser.add_argument('--output',
                        dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='desired output directory')
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

    compress_dataset(args.dataset_path, args.output_dir, args.height,
                     args.width)

    sys.stdout.write("\033")
    print("Succesfully compressed dataset to %s with no errors." %
          args.output_dir)
    return 0


def compress_dataset(dataset_path,
                     output_dir=DEFAULT_OUTPUT_DIR,
                     height=DEFAULT_HEIGHT,
                     width=DEFAULT_WIDTH,
                     overwrite=DEFAULT_OVERWRITE):
    """Compresses the PRImA layout analysis dataset to a specified path.

    Arguments:
        dataset_path: Path to the dataset. The dataset must contain a
            subdirectory named "XML" and a subdirectory named "Images".
        output_dir: The desired path for the compressed dataset.
            The directory will be created if it does not currently exist.
        height: The height to which the images will be compressed to.
        width: The width to which the images will be compressed to.
        overwrite: If true, then duplicate files will be overwritten.
    """
    make_output_directories(output_dir)
    compress_images(dataset_path, output_dir, height, width, overwrite)


def make_output_directories(output_dir):
    """Creates output directory for the compressed dataset.

    Arguments:
        output_dir: A path to the desired output directory location
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def update_filename(image_filename):
    """Updates the extension on an image filename.

    This method accepts both paths to images and filename.

    Arguments:
        image_filename: Either a path to an image or an iamge filename

    Returns:
        the image filename with the proper extension. If a path is passed as an
        argument, then the path is removed and only the filename is returned.
    """
    return get_filename_without_extension(image_filename) + IMAGE_EXTENSION


def get_filename_without_extension(path):
    """Given a path to a file, returns the filename without the extension."""
    # This gets the filename from the path
    path_head = os.path.split(path)[1]
    # This replaces the old file extension with the new extension (e.g, "jpeg")
    return os.path.splitext(path_head)[0]


def compress_images(image_dir,
                    output_dir=DEFAULT_OUTPUT_DIR,
                    height=DEFAULT_HEIGHT,
                    width=DEFAULT_WIDTH,
                    overwrite=DEFAULT_OVERWRITE):
    """Compress all images in the specified directory.

    Arguments:
        image_dir: Path to the image directory.
        output_dir: The desired path for the compressed dataset.
        height: The height to which the images will be compressed to.
        width: The width to which the images will be compressed to.
        overwrite: If true, then duplicate files will be overwritten.
    """
    image_filenames = os.listdir(image_dir)
    num_images = len(image_filenames)

    for num_images_compressed, filename in enumerate(image_filenames):
        image_path = image_dir + "/" + filename
        compress_image(image_path, output_dir, height, width, overwrite)

        percent = "{0:.0%}".format(num_images_compressed / num_images)
        print("Compressing images | %s | %s" % (percent, filename), end="\r")


def compress_image(image_path,
                   output_dir,
                   height=DEFAULT_HEIGHT,
                   width=DEFAULT_WIDTH,
                   overwrite=DEFAULT_OVERWRITE):
    """Compresses the image at the specified file path.

    The image will be converted to JPEG and resized to the specified resolution.
    The name of the image is preserved, with the slight exception of changes to
    the file extension.

    Arguments:
        image_path: Path to an image.
        output_dir: The compressed image will be saved here.
        height: The height to which the images will be compressed to.
        width: The width to which the images will be compressed to.
        overwrite: If true, then duplicate files will be overwritten.
    """
    output_path = output_dir + "/" + update_filename(image_path)

    if os.path.exists(output_path) and not overwrite:
        print("Warning: %s already exists. Image will be skipped." %
              output_path)
        return

    image = Image.open(image_path)
    desired_resolution = width, height
    resized_image = image.resize(desired_resolution)
    rgb_image = resized_image.convert("RGB")

    rgb_image.save(output_path, format="PNG", optimized=True, quality=75)


if __name__ == "__main__":
    main()
