import torch
from torchvision.transforms import v2

"""
Provides data transformation and augmentation utilities for images and bounding boxes.

Usage example:

image = ...
boxes = ...

# image and boxes must be tv_tensor types (tv_tensor.Image and tv_tensor.BoundingBox)

transform = Transform()
image, boxes = transform(image, boxes)
"""


class Augment:
    """
    Data Augmentation class.

    This class applies data augmentation transformations to images and bounding boxes with a
    specified probability.
    Expected inputs are tv_tensor.Image and tv_tensor.BoundingBox objects.
    Outputs are tensors.
    """

    def __init__(self):
        self.prob = 0.5

    def __call__(self, *args, **kwargs):
        transform_list = [
            v2.RandomPhotometricDistort(p=self.prob),
            v2.ColorJitter(brightness=0.126, saturation=0.5),
            v2.RandomAffine(
                degrees=(-15, 15),
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                fill=0,
            ),
        ]

        transform = v2.Compose(
            [
                v2.RandomApply(transform_list, p=0.5),
                v2.ToPureTensor(),
            ]
        )

        return transform(*args, **kwargs)


class Transform:
    """
    Image transformation class.

    This class transforms an image into a tensor with values scaled to the range [0,1].

    Example usage:
        img = PIL.Image.open(image_path).convert("RGB")
        transform = Transform()
        img = transform(img)
    """

    def __call__(self, *args, **kwargs):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.ToPureTensor(),
            ]
        )

        return transform(*args, **kwargs)
