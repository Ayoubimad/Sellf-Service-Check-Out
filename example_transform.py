import supervision
import torch

from datasets import FoodBinary
from encoders import LabelEncoder
from transform import Augment
from utils import annotate

##################################################################################################

# Data Augmentation food101 binary dataset
encoder = LabelEncoder(["food"])

json_file = "datasets/food-101/meta/food101_train_new_annotations.json"
image_dir = "datasets/food-101/filtred_images/"

dataset = FoodBinary(
    json_file=json_file,
    image_dir=image_dir,
    encoder=encoder,
    augment=Augment()
)

for i in range(5):

  image, target = dataset[i]

  image = torch.permute(image, (1, 2, 0))  # CxHxW --> HxWxC

  image = image.numpy()

  img = annotate(
      image,
      boxes=target["boxes"].numpy(),
      class_id=target["labels"].numpy(),
      phrases=[encoder.get_label(l) for l in target["labels"]],
      denormalize=False,
  )

  supervision.plot_image(img, (8, 8))
