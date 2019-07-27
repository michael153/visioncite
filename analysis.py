"""This module implements function for analyzing models and their predictions."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize(image, mask, classes):
    # torch.Size([C, H, W]) => torch.Size([H, W, C])
    image = image.permute(1, 2, 0)
    # This is necessary so that colors are intepreted correctly
    image = image.int()

    plt.imshow(image)

    mask_image = plt.imshow(mask, vmin=0, vmax=len(classes) - 1, alpha=0.85)

    num_classes = len(classes)
    colors = [
        mask_image.cmap(mask_image.norm(value)) for value in range(num_classes)
    ]
    patches = [
        mpatches.Patch(color=colors[i], label=classes[i])
        for i in range(num_classes)
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2)

    plt.show()
