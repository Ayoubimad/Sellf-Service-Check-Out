import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

"""
Fine-tuned Faster-RCNN implementation for object detection.

Reference: @https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""


class Faster_RCNN:
    """
    Class for fine-tuning and using a Faster R-CNN model with ResNet50 backbone.

    Attributes:
        num_classes (int): The number of classes for the detection model.
        pretrained (bool): If True, loads the model with COCO-pretrained weights.
        model (torch.nn.Module): The Faster R-CNN model for object detection.
    """

    def __init__(self, num_classes=2, pretrained=True):
        """
        Initializes  Faster RCNN model.

        Args:
            num_classes (int): The number of classes for detection, including background.
            pretrained (bool): If True, initializes model with COCO-pretrained weights.
        """
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = self._get_model()

    def __call__(self, *args, **kwargs):
        """
        Makes the class instance callable, allowing inference on inputs.

        Args:
            *args, **kwargs: Inputs to be passed to the model during inference.

        Returns:
            Model predictions for the given inputs.
        """
        model = self.get_model()
        return model(*args, **kwargs)

    def _get_model(self):
        """
        Initializes the Faster R-CNN model with or without pretrained weights and customizes
        the prediction layer to match the specified number of classes.

        Returns:
            torch.nn.Module: The initialized Faster R-CNN model.
        """
        if self.pretrained:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        # Replace the model's box predictor to match the specified number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model

    def get_model(self):
        """
        Returns the model instance.

        Returns:
            torch.nn.Module: The Faster R-CNN model.
        """
        return self.model

    def to(self, device):
        """
        Moves the model to the specified device.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.get_model().to(device=device)

    def parameters(self):
        """
        Returns the parameters of the model, useful for setting up optimizers.

        Returns:
            iterator: Model parameters.
        """
        return self.get_model().parameters()

    def predict(
        self,
        image: torch.Tensor,
        device: torch.device = "cpu",
        score_threshold: float = 0.8,
    ) -> dict:
        """
        Performs inference on an input image and filters the predictions based on a confidence threshold.

        Args:
            image (torch.Tensor): The input image tensor normalized to [0,1].
            device (torch.device): The device to perform inference on.
            score_threshold (float): Minimum confidence score to retain predictions.

        Returns:
            dict: Filtered predictions containing bounding boxes, labels, and scores.
        """
        self.model.eval()
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = self.model(image)

        prediction = prediction[0]
        high_score_indices = prediction["scores"] > score_threshold

        filtered_prediction = {
            "boxes": prediction["boxes"][high_score_indices],
            "labels": prediction["labels"][high_score_indices],
            "scores": prediction["scores"][high_score_indices],
        }

        return filtered_prediction

    def load_weights(self, path: str) -> None:
        """
        Loads model weights from a specified file path.

        Args:
            path (str): Path to the weights file.
        """
        model_state_dict = torch.load(path, map_location="cpu")
        self.get_model().load_state_dict(model_state_dict)

    def train(self):
        """
        Sets the model in training mode. In training mode, the model returns losses.

        Returns:
            torch.nn.Module: The model in training mode.
        """
        return self.get_model().train()

    def eval(self):
        """
        Sets the model in evaluation mode. In evaluation mode, the model returns predictions.

        Returns:
            torch.nn.Module: The model in evaluation mode.
        """
        return self.get_model().eval()

    def state_dict(self):
        """
        Returns the state dictionary of the model.

        Returns:
            dict: The model's state dictionary.
        """
        return self.get_model().state_dict()
