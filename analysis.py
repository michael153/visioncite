"""This module implements function for analyzing models and their predictions."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets import PRImADataset


def visualize(mask, image=None):
    """Shows a visualization on an image mask.

    Some terminology: H is the height of an image in pixels; W is the width of
    an image in pixels; C is the number of channels; N is the number of classes.

    This function assumes that the PRImA dataset is being used.

    Arguments:
        mask: A tensor of shape [H, W] containing elements in {0, ..., N}.
        image: A tensor of shape [C, H, W].

    Example:
        >>> model = CNN(len(PRImADataset.CLASSES))
        >>> model.load_state_dict(torch.load("662150.b64e256.model"))
        >>>
        >>> dataset = PRImADataset("small.train", "images", "labels")
        >>> # We're arbitrarily choosing the first image in the dataset
        >>> image = dataset[0]["image"]
        >>> # We need to reshape the input so that the model will accept it
        >>> batch = torch.stack([sample])
        >>>
        >>> prediction = model(batch)
        >>> mask = prediction_to_mask(prediction)
        >>> visualize(mask, image)
    """
    # torch.Size([3, 384, 256]) => torch.Size([384, 256, 3])
    image = image.permute(1, 2, 0)
    # This is necessary so that colors are intepreted correctly
    image = image.int()

    opacity = 1 if image is None else 0.85

    plt.imshow(image)
    mask_image = plt.imshow(mask, vmin=0, vmax=12, alpha=opacity)

    num_classes = len(PRImADataset.CLASSES)
    colors = [
        mask_image.cmap(mask_image.norm(value)) for value in range(num_classes)
    ]
    patches = [
        mpatches.Patch(color=colors[i], label=PRImADataset.CLASSES[i])
        for i in range(num_classes)
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2)

    plt.show()


def prediction_to_mask(prediction):
    """Converts a tensor outputted by a CNN to an image mask"""
    # torch.Size([1, 12, 384, 256]) => torch.Size([12, 384, 256])
    prediction = prediction.squeeze()
    # torch.Size([12, 384, 256]) => torch.Size([384, 256])
    mask = prediction.argmax(dim=0)
    return mask
