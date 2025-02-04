from torch.utils.data import DataLoader
from config import Config
from encoders import LabelEncoder
from datasets import FoodBinary
from trainer_faster_rcnn import FasterRCNNTrainer
from transform import Transform, Augment
from torch.utils.data import DataLoader

config = Config()

encoder = LabelEncoder(config.encoder_classes)

train_dataset_food_101 = FoodBinary(
    json_file=config.json_train_file,
    image_dir=config.image_dir,
    encoder=encoder,
    transform=Transform(),
    augment=Augment(),
)

train_loader_food_101 = DataLoader(
    train_dataset_food_101,
    batch_size=config.batch_size,
    collate_fn=lambda x: tuple(zip(*x)),
)

valid_dataset_food_101 = FoodBinary(
    json_file=config.json_valid_file,
    image_dir=config.image_dir,
    encoder=encoder,
    transform=Transform(),
)

valid_loader_food_101 = DataLoader(
    valid_dataset_food_101,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x)),
)

trainer = FasterRCNNTrainer(
    config,
)

trainer.train(train_loader=train_loader_food_101, valid_loader=valid_loader_food_101)
