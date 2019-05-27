import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class PRImADataset(Dataset):
    """Abstraction for the PRImA Layout Analysis Dataset."""

    CLASSES = [
        'other',
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
    ]

    def __init__(self, data_file, image_dir, label_dir, transform=None):
        """Initializes a PRImADataset object.

        The image directory and label directory should correspond to data of
        matching dimensions.

        The train file should contain newline-separated label names. Label names
        should not include paths. For example, write "00000085.xml" rather than
        "images-384x256/00000085.xml".

        Arguments:
            data_file: A path to a train file (e.g, ./32020191045.train).
            image_dir: The path to the dataset images (e.g, ./images-384x256).
            label_dir: The path to the dataset labels (e.g, ./labels-384x256).
            transform: Does nothing for now. I might implement this later.
        """
        with open(data_file) as file:
            self.image_names = file.readlines()
        self.transform = transform

    def __getitem__(self, index):

        def import_image(image_path):
            image = Image.open(image_path)
            image_array = np.asarray(image)
            image_tensor = torch.tensor(image_array)
            return image_tensor

        image_filename = image_names[index]
        image_path = os.path.join(image_dir, image_filename)
        image = import_image(image_path)

        label_dom = xml_to_json()
        mask = None

        sample = { "image": image, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(image_names)


def show_mask(image, mask):
    plt.imshow(image)


def show_batch(sample_batched):
    pass
