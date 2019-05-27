"""Implements functions for compressing the PRImA layout analysis dataset."""
import argparse
import os
import sys
import math
from xml.dom import minidom

from PIL import Image

DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 256
DEFAULT_OUTPUT_DIR = "dataset-%s-%s" % (DEFAULT_HEIGHT, DEFAULT_WIDTH)
DEFAULT_OVERWRITE = False

IMAGE_EXTENSION = ".jpeg"


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

    if not is_valid(args.dataset_path):
        print("Error: dataset must exist and contain a subdirectory Images" +
              "and a subdirectory named XML")
        return 1

    compress_dataset(args.dataset_path, args.output_dir, args.height,
                     args.width)

    sys.stdout.write("\033")
    print("Succesfully compressed dataset to %s with no errors." %
          args.output_dir)
    return 0


def is_valid(dataset_path):
    """Returns true if the specified dataset is valid."""
    return os.path.exists(dataset_path) and os.path.exists(
        dataset_path + "/Images") and os.path.exists(dataset_path + "/XML")


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
    update_labels(dataset_path + "/XML", output_dir + "/XML", height, width,
                  overwrite)
    compress_images(dataset_path + "/Images", output_dir + "/Images", height,
                    width, overwrite)
    remove_mismatches(output_dir + "/Images", output_dir + "/XML")


def make_output_directories(output_dir):
    """Creates output directory for the compressed dataset.

    Subdirectories named "json" and "images" will also be created.

    Arguments:
        output_dir: A path to the desired output directory location
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + "/Images"):
        os.makedirs(output_dir + "/Images")
    if not os.path.exists(output_dir + "/XML"):
        os.makedirs(output_dir + "/XML")


def update_labels(label_dir,
                  output_dir=DEFAULT_OUTPUT_DIR,
                  height=DEFAULT_HEIGHT,
                  width=DEFAULT_WIDTH,
                  overwrite=DEFAULT_OVERWRITE):
    """Resize and update metadata for all labels in the specified directory.

    Arguments:
        label_dir: Path to the label directory.
        output_dir: The updated label file will be saved here.
        height: The height to which the images will be compressed to.
        width: The width to which the images will be compressed to.
        overwrite: If true, then duplicate files will be overwritten.

    Returns:
        The set of extensionless filenames corresponding to the XML files that
        errored.
    """
    label_filenames = os.listdir(label_dir)
    num_labels = len(label_filenames)
    for num_labels_resized, filename in enumerate(label_filenames):
        label_path = label_dir + "/" + filename

        try:
            update_label(label_path, output_dir, height, width, overwrite)
        except CompressionError:
            print("Warning: " + label_path +
                  " contained an error label will be skipped.")

        percent = "{0:.0%}".format(num_labels_resized / num_labels)
        print("Resizing labels | %s | %s" % (percent, filename), end="\r")


def update_label(label_path,
                 output_dir,
                 height=DEFAULT_HEIGHT,
                 width=DEFAULT_WIDTH,
                 overwrite=DEFAULT_OVERWRITE):
    """Resize and update metadata for a specific XML label.

    Arguments:
        label_path: Path to an XML label.
        output_dir: The compressed image will be saved here.
        height: The height to which the images will be compressed to.
        width: The width to which the images will be compressed to.
        overwrite: If true, then duplicate files will be overwritten.
    """
    # This gets the filename from the path
    filename = os.path.split(label_path)[1]
    output_path = output_dir + "/" + filename

    if os.path.exists(output_path) and not overwrite:
        print("Warning: %s already exists. Label will be skipped." %
              output_path)
        return

    label_dom = minidom.parse(label_path)

    x_scale, y_scale = get_scaling_functions(label_dom, height, width)
    scale_point_data(label_dom, x_scale, y_scale)
    update_metadata(label_dom, height, width)

    with open(output_path, "w+" if overwrite else "w") as file:
        label_dom.writexml(file)


def update_metadata(label_dom, height, width):
    """Updates the metadata part of an XML label."""
    page_elements = label_dom.getElementsByTagName("Page")
    if len(page_elements) != 1:
        # This seems to happen when the label is written in lower case
        raise CompressionError("Incorrect number of page elements.")
    page_element = page_elements[0]

    original_filename = page_element.getAttribute('imageFilename')
    new_filename = update_filename(original_filename)

    page_element.setAttribute('imageFilename', new_filename)
    page_element.setAttribute('imageHeight', str(height))
    page_element.setAttribute('imageWidth', str(width))


def get_scaling_functions(label_dom, height, width):
    """Returns a pair of one-argument functions that scale point values."""
    page_elements = label_dom.getElementsByTagName("Page")
    if len(page_elements) != 1:
        # This seems to happen when the label is written in lower case
        raise CompressionError("Incorrect number of page elements.")
    page_element = page_elements[0]

    original_height = int(page_element.getAttribute('imageHeight'))
    original_width = int(page_element.getAttribute('imageWidth'))
    x_scale = lambda x: math.floor(x * width / original_width)
    y_scale = lambda y: math.floor(y * height / original_height)

    return x_scale, y_scale


def scale_point_data(label_dom, x_scale, y_scale):
    """Scales points to match new image size."""
    region_types = {"TextRegion", "ImageRegion", "GraphicRegion"}
    for region_type in region_types:
        regions = label_dom.getElementsByTagName(region_type)
        for region in regions:

            coordinates_elements = region.getElementsByTagName("Coords")
            if len(coordinates_elements) != 1:
                raise CompressionError("Incorrect number of coordinates.")
            coordinates = coordinates_elements[0]

            points = coordinates.getElementsByTagName("Point")

            if not points:
                # This might also happen when the label has incorrect casing
                raise CompressionError("No points found.")

            for point in points:
                x_value = int(point.getAttribute("x"))
                x_value = x_scale(x_value)
                point.setAttribute("x", str(x_value))

                y_value = int(point.getAttribute("y"))
                y_value = y_scale(y_value)
                point.setAttribute("y", str(y_value))


class CompressionError(Exception):
    """Exception representing errors that occur during compression."""


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

    rgb_image.save(output_path, format="JPEG", optimized=True, quality=75)


def remove_mismatches(image_dir, xml_dir):
    """Removes images that do not have corresponding labels.

    This functions assumes that all labels are XML files, and that image-label
    pairs will share the same extensionless filename.
    """
    for image_filename in os.listdir(image_dir):
        extensionless_filename = get_filename_without_extension(image_filename)
        identifier = get_identifier(extensionless_filename)
        if not contains_identifier(xml_dir, identifier):
            image_path = image_dir + "/" + image_filename
            print("Warning: Cannot find label for %s. Image will be removed" %
                  image_filename)
            os.remove(image_path)


def contains_identifier(directory, target):
    """Returns true if the directory contains the target identifier.

    Go to the get_identifier function to see how identifier is defined.
    """
    for filename in os.listdir(directory):
        extensionless_filename = get_filename_without_extension(filename)
        identifier = get_identifier(extensionless_filename)
        if identifier == target:
            return True
    return False


def get_identifier(extensionless_filename):
    """Returns the four digits at the end of a filename."""
    return int(extensionless_filename[-4:]) if extensionless_filename else 0

if __name__ == "__main__":
    main()
