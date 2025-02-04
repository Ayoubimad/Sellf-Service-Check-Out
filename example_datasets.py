import numpy as np
import scipy
import supervision as sv
import torch
from torch.utils.data import DataLoader

from datasets import UNIBIMDataset
from encoders import LabelEncoderUNIMIB2016
from utils import annotate

if __name__ == "__main__":

    UNIMIB_JSON_FILE = "./datasets/UNIBIM2016/UNIMIB2016-annotations/annotations.json"
    UNIMIB_IMAGE_DIR = "./datasets/UNIBIM2016/original"
    UNIMIB_MAT_FILE = "./datasets/UNIBIM2016/split/final_food_list.mat"
    BATCH_SIZE = 8

    class_list = scipy.io.loadmat(UNIMIB_MAT_FILE)["final_food_list"]
    class_list = [str(item[0][0]) for item in class_list]

    encoder_unimib = LabelEncoderUNIMIB2016(class_list)

    dataset_unimib = UNIBIMDataset(
        json_file=UNIMIB_JSON_FILE, image_dir=UNIMIB_IMAGE_DIR, encoder=encoder_unimib
    )

    train_loader = DataLoader(
        dataset=dataset_unimib,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    for images, target in train_loader:

        for i in range(len(images)):

            img = images[i]

            torch.clamp(img, 0, 255)

            img = img.permute(1, 2, 0).numpy()

            phrases = [encoder_unimib.get_label(l) for l in target[i]["labels"]]

            img = annotate(
                image_source=img,
                boxes=target[i]["boxes"].numpy(),
                confidence=np.ones(len(target[i]["boxes"])),
                class_id=target[i]["labels"].numpy(),
                phrases=phrases,
            )

            sv.plot_image(img, (8, 8))

        break
