import json
import os

import PIL
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import ImageDraw
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

from encoders import LabelEncoder, LabelEncoderUNIMIB2016
from transform import Transform

"""
Reference: @ https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

"""

# okok


class Food101(Dataset):

    def __init__(
        self, file_path: str, img_dir_path: str, encoder: LabelEncoder, transform=None
    ):
        """
        Initializes the Food101 dataset.

        :param file_path: Path to the file containing image names
                          (e.g., 'apple_pie/image_name.jpg', 'sushi/image_name.jpg', etc.).
                          Example files:
                          - /datasets/food-101/meta/train.txt
                          - /datasets/food-101/meta/test.txt
        :param encoder: LabelEncoder instance to encode labels.
        :param img_dir_path: Path to the directory containing images.
                            (e.g. , /datasets/food-101/images/
        :param transform: Transformations to apply to the images (optional).
        """

        self.file_path = file_path
        self.img_dir_path = img_dir_path
        self.encoder = encoder
        self.transform = transform if transform is not None else Transform()

        self.resize = v2.Compose(
            [
                v2.Resize((512, 512)),
            ]
        )

        self.dataframe = self._prepare_dataframe(self.file_path)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        :return: Integer representing the number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, index: int):
        """
        Returns a single sample from the dataset.

        :param index: Index of the sample to return.
        :return: A tuple containing an image tensor (CxHxW) normalized [0,1] and a label (index).
        """
        img_name = self.dataframe.iloc[index]["path"]
        image = self._load_image(img_name)
        label = self.encoder.get_index(self.dataframe.iloc[index]["label"])
        return image, label

    def _load_image(self, img_name: str) -> torch.Tensor:
        """
        Loads an image and applies the necessary transformations.

        :param img_name: Path to the image file.
        :return: Transformed image tensor.
        """
        image = Image.open(img_name)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
            image = self.resize(image)
        return image

    def _prepare_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Prepares a dataframe containing image paths and labels.

        :param file_path: Path to the file containing image information.
        :return: A Pandas DataFrame with image paths and labels.
        """
        with open(file_path, "r") as file:
            lines = file.read().splitlines()

        data = [
            {
                "label": line.split("/")[0],
                "path": os.path.join(self.img_dir_path, f"{line}.jpg"),
            }
            for line in lines
        ]
        df = pd.DataFrame(data)
        return shuffle(df)


class FoodBinary(Dataset):

    def __init__(
        self,
        json_file: str,
        image_dir: str,
        encoder: LabelEncoder,
        transform=None,
        augment=None,
    ):
        """
        Initializes the Food101 dataset with a single 'food' class.
        Note: Bounding boxes are normalized [0,1].

        :param json_file: Path to the JSON file containing annotations.
        :param image_dir: Directory containing the images.
        :param encoder: LabelEncoder instance to encode labels.
        :param transform: Transformations to apply to the images (optional).
        :param augment: Augmentation function to apply to the images and targets (optional).
        """
        self.encoder = encoder
        self.image_dir = image_dir
        self.transform = transform if transform is not None else Transform()
        self.augment = augment
        self.data = self._load_data(json_file)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns a single sample from the dataset.

        :param idx: Index of the sample to return.
        :return: A tuple containing a transformed image and a dictionary with bounding boxes and labels.
        """
        image_info = self.data[idx]

        image = self._load_image(image_info["image_name"])

        img_width, img_height = image.size

        if self.transform:
            image = self.transform(image)

        boxes = self._process_boxes(image_info["bboxes"], img_width, img_height)
        labels = self._encode_labels(image_info["labels"])

        target = {"boxes": boxes, "labels": labels}

        if self.augment is not None:
            image = tv_tensors.Image(image)
            target["boxes"] = tv_tensors.BoundingBoxes(
                target["boxes"], format="XYXY", canvas_size=image.shape[-2:]
            )

            image, target = self.augment(image, target)

            valid_boxes = []
            valid_labels = []
            for box, label in zip(target["boxes"], target["labels"]):
                if (box[2] - box[0] > 0) and (box[3] - box[1] > 0):
                    valid_boxes.append(box)
                    valid_labels.append(label)

            if not valid_boxes:
                return self.__getitem__((idx + 1) % len(self)) # Brutto ma per ora va

            target["boxes"] = torch.stack(valid_boxes)
            target["labels"] = torch.tensor(valid_labels, dtype=torch.int64)

        return image, target

    def _load_data(self, json_file: str) -> list:
        """
        Loads and filters the data from the JSON file.

        :param json_file: Path to the JSON file containing annotations.
        :return: List of annotations with bounding boxes.
        """
        with open(json_file, "r") as f:
            data = json.load(f)
        return [item for item in data if item["bboxes"]]

    def _load_image(self, image_name: str) -> Image.Image:
        """
        Loads and converts the image to RGB.

        :param image_name: Name of the image file.
        :return: PIL Image in RGB mode.
        """
        img_path = os.path.join(self.image_dir, image_name)
        return Image.open(img_path).convert("RGB")

    def _process_boxes(
        self, bboxes: list, img_width: int, img_height: int
    ) -> torch.Tensor:
        """
        Processes bounding boxes by converting and denormalizing them.
        Also validates and fixes invalid boxes.

        :param bboxes: List of bounding boxes.
        :param img_width: Width of the image.
        :param img_height: Height of the image.
        :return: Tensor of processed bounding boxes in [xmin, ymin, xmax, ymax] format.
        """
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes *= torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)

        return boxes

    def _encode_labels(self, labels: list) -> torch.Tensor:
        """
        Encodes labels using the provided LabelEncoder.

        :param labels: List of labels.
        :return: Tensor of encoded labels.
        """
        encoded_labels = [self.encoder.get_index(label) for label in labels]
        return torch.tensor(encoded_labels, dtype=torch.int64)


class UNIBIMDataset(Dataset):

    def __init__(
        self,
        json_file: str,
        image_dir: str,
        encoder: LabelEncoderUNIMIB2016,
        transform=None,
    ):
        """
        Initializes the UNIBIM dataset.

        :param json_file: Path to the JSON file containing annotations.
                          Example: datasets/UNIBIM2016/UNIMIB2016-annotations/annotations.json
        :param image_dir: Directory containing the images.
                          Example: datasets/UNIBIM2016/original
        :param encoder: LabelEncoder instance to encode labels.
        :param transform: Transformations to apply to the images (optional).
        """
        self.image_dir = image_dir
        self.transform = transform if transform is not None else Transform()
        self.encoder = encoder
        self.data = self._load_annotations(json_file)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns a single sample from the dataset.

        :param idx: Index of the sample to return.
        :return: A tuple containing a transformed image and a dictionary with bounding boxes, labels, and masks.
        """
        image_info = self.data[idx]
        image = self._load_image(image_info["image_name"])
        target = self._get_target(image_info["annotations"], image.size)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def _load_annotations(self, json_file: str) -> list:
        """
        Loads annotations from the JSON file.

        :param json_file: Path to the JSON file containing annotations.
        :return: List of annotations.
        """
        with open(json_file, "r") as f:
            return json.load(f)["annotations"]

    def _load_image(self, image_name: str) -> Image.Image:
        """
        Loads an image from the dataset.

        :param image_name: Name of the image file.
        :return: PIL Image object.
        """
        img_path = os.path.join(self.image_dir, image_name + ".jpg")
        return Image.open(img_path).convert("RGB")

    def _get_target(self, annotations: list, image_size: tuple[int, int]) -> dict:
        """
        Processes annotations and creates the target dictionary.

        :param annotations: List of annotations for the image.
        :param image_size: Size of the image.
        :return: Dictionary containing bounding boxes, labels, and masks.
        """
        boxes = []
        labels = []

        for ann in annotations:
            labels.append(self.class_to_label(ann["class"]))
            boxes.append(self._convert_bbox(ann["bounding_box"]))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return target

    def class_to_label(self, class_name: str) -> int:
        """
        Converts a class name into its corresponding label index.

        :param class_name: Name of the class.
        :return: Integer representing the class index.
        """
        return self.encoder.get_index(class_name)

    def _convert_bbox(self, bbox: list[float]) -> list[float]:
        """
        Converts bounding box coordinates to [xmin, ymin, xmax, ymax] format.

        :param bbox: List containing bounding box coordinates.
        :return: List of floats representing the converted bounding box.
        """
        xmin = min(bbox[0::2])
        ymin = min(bbox[1::2])
        xmax = max(bbox[0::2])
        ymax = max(bbox[1::2])
        return [xmin, ymin, xmax, ymax]

    def _create_mask(
        self, boundary_points: list[float], image_size: tuple[int, int]
    ) -> np.ndarray:
        """
        Creates a binary mask for the object based on boundary points.

        :param boundary_points: List of points [x1, y1, x2, y2, ...] defining the object's boundary.
        :param image_size: Tuple containing the image size (width, height).
        :return: NumPy array representing the binary mask.
        """
        mask = Image.new("L", image_size, 0)  # 'L' for grayscale image
        draw = ImageDraw.Draw(mask)
        polygon = [
            (boundary_points[i], boundary_points[i + 1])
            for i in range(0, len(boundary_points), 2)
        ]
        draw.polygon(polygon, outline=1, fill=1)
        return np.array(mask)


class UNIBIMBinary(Dataset):

    def __init__(
        self,
        json_file: str,
        image_dir: str,
        encoder: LabelEncoder,
        transform=None,
        augment=None,
    ):
        """
        Initializes the UNIBIM dataset for binary detection.

        :param json_file: Path to the JSON file containing annotations.
                          Example: datasets/UNIBIM2016/UNIMIB2016-annotations/annotations.json
        :param image_dir: Directory containing the images.
                          Example: datasets/UNIBIM2016/original
        :param encoder: LabelEncoder instance to encode labels.
        :param transform: Transformations to apply to the images (optional).
        """
        self.image_dir = image_dir
        self.encoder = encoder
        self.transform = transform if transform is not None else Transform()
        self.data = self._load_annotations(json_file)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns a single sample from the dataset.

        :param idx: Index of the sample to return.
        :return: A tuple containing a transformed image and a dictionary with bounding boxes and labels.
        """
        image_info = self.data[idx]
        image = self._load_image(image_info["image_name"])
        image = self.transform(image)
        target = self._process_annotations(image_info["annotations"])

        return image, target

    def _load_annotations(self, json_file: str) -> list:
        """
        Loads annotations from the JSON file.

        :param json_file: Path to the JSON file containing annotations.
        :return: List of annotations.
        """
        with open(json_file, "r") as f:
            return json.load(f)["annotations"]

    def _load_image(self, image_name: str) -> PIL.Image:
        """
        Loads an image from the dataset.

        :param image_name: Name of the image file.
        :return: PIL Image in RGB mode.
        """
        img_path = os.path.join(self.image_dir, image_name + ".jpg")
        return Image.open(img_path).convert("RGB")

    def _process_annotations(self, annotations: list) -> dict:
        """
        Processes annotations to extract bounding boxes and labels.

        :param annotations: List of annotations for the image.
        :return: A tuple containing tensors for bounding boxes and labels.
        """
        boxes = []
        labels = []

        for ann in annotations:
            labels.append(self.class_to_label("food"))
            boxes.append(self._convert_bbox(ann["bounding_box"]))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        return target

    def class_to_label(self, class_name: str) -> int:
        """
        Converts a class name into its corresponding label index.

        :param class_name: Name of the class.
        :return: Integer representing the class index.
        """
        return self.encoder.get_index(class_name)

    def _convert_bbox(self, bbox: list[float]) -> list[float]:
        """
        Converts bounding box coordinates to [xmin, ymin, xmax, ymax] format.

        :param bbox: List containing bounding box coordinates.
        :return: List of floats representing the converted bounding box.
        """
        xmin = min(bbox[0::2])
        ymin = min(bbox[1::2])
        xmax = max(bbox[0::2])
        ymax = max(bbox[1::2])
        return [xmin, ymin, xmax, ymax]

    def _create_mask(
        self, boundary_points: list[float], image_size: tuple[int, int]
    ) -> np.ndarray:
        """
        Creates a binary mask for the object based on boundary points.

        :param boundary_points: List of points [x1, y1, x2, y2, ...] defining the object's boundary.
        :param image_size: Tuple containing the image size (width, height).
        :return: NumPy array representing the binary mask.
        """
        mask = Image.new("L", image_size, 0)  # 'L' for grayscale image
        draw = ImageDraw.Draw(mask)
        polygon = [
            (boundary_points[i], boundary_points[i + 1])
            for i in range(0, len(boundary_points), 2)
        ]
        draw.polygon(polygon, outline=1, fill=1)
        return np.array(mask)
