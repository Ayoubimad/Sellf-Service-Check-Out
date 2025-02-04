import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from config import Config
from encoders import LabelEncoder
from datasets import FoodBinary
from transform import Transform
from faster_rcnn import Faster_RCNN
from torchmetrics.detection.mean_ap import MeanAveragePrecision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
encoder = LabelEncoder(config.encoder_classes)

test_dataset_food_101 = FoodBinary(
    json_file=config.json_test_file,
    image_dir=config.image_dir,
    encoder=encoder,
    transform=Transform(),
)

test_loader_food_101 = DataLoader(
    test_dataset_food_101,
    batch_size=config.batch_size,
    collate_fn=lambda x: tuple(zip(*x)),
)

model = Faster_RCNN()
model.load_weights("./weights/faster_rcnn.pth")
model.to(device)
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for images, targets in tqdm(test_loader_food_101, desc="Evaluating model"):

        images = [img.to(device) for img in images]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

metric = MeanAveragePrecision()
metric.update(all_predictions, all_targets)
results = metric.compute()

print("\nModel Evaluation Metrics:")
print(f"Average Precision: {results['map'].item():.4f}")
print(f"Average Precision (IoU=0.50): {results['map_50'].item():.4f}")
print(f"Average Precision (IoU=0.75): {results['map_75'].item():.4f}")
