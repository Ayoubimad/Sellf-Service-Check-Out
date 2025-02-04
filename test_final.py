import torch
import json
import os
import scipy
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

from siamese import SiameseNetwork
from datasets import UNIBIMDataset
from encoders import LabelEncoderUNIMIB2016
from faster_rcnn import Faster_RCNN
from torchvision import transforms

UNIMIB_JSON_FILE = (
    "./datasets/UNIBIM2016/UNIMIB2016-annotations/annotations_cleaned.json"
)
UNIMIB_IMAGE_DIR = "./datasets/UNIBIM2016/original"
UNIMIB_MAT_FILE = "./datasets/UNIBIM2016/split/final_food_list.mat"
SIAMESE_MODEL_PATH = "./weights/siamese.pth"
FASTER_RCNN_MODEL_PATH = "./weights/faster_rcnn.pth"
BATCH_SIZE = 8

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

normalize_transform = transforms.Compose(
    [transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)]
)

# Debug
SHOW_PLOTS = False
PLOT_COMPARISONS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_list = scipy.io.loadmat(UNIMIB_MAT_FILE)["final_food_list"]
class_list = [str(item[0][0]) for item in class_list]

encoder_unimib = LabelEncoderUNIMIB2016(class_list)
dataset_unimib = UNIBIMDataset(
    json_file=UNIMIB_JSON_FILE, image_dir=UNIMIB_IMAGE_DIR, encoder=encoder_unimib
)

test_loader = DataLoader(
    dataset=dataset_unimib,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    collate_fn=lambda x: tuple(zip(*x)),
)

detector = Faster_RCNN()
detector.load_weights(FASTER_RCNN_MODEL_PATH)
detector.to(DEVICE)
detector.eval()

feature_extractor = SiameseNetwork()
feature_extractor.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location="cpu"))
feature_extractor.to(DEVICE)
feature_extractor.eval()


def plot_roi(roi_tensor):
    """Plots a single ROI tensor or Image."""
    plt.figure(figsize=(5, 5))
    plt.imshow(roi_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.show()


def apply_bilateral_filter(roi_tensor):
    """Applies bilateral filter to ROI tensor."""
    roi_np = roi_tensor.cpu().numpy().transpose(1, 2, 0)
    roi_np = (roi_np * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(roi_np, d=9, sigmaColor=75, sigmaSpace=75)
    return torch.from_numpy(filtered.transpose(2, 0, 1)).float() / 255.0


def plot_comparison(pred_roi, gt_class_roi, pred_class, gt_class):
    """Plots comparison between predicted and ground truth ROIs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(pred_roi.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax1.set_title(f"Predicted: {pred_class}")
    ax1.axis("off")

    gt_roi = torch.tensor(gt_class_roi)
    ax2.imshow(gt_roi.permute(1, 2, 0).cpu().numpy())
    ax2.set_title(f"Ground Truth: {gt_class}")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def calculate_iou(box1, box2):
    """Calculates Intersection over Union between two boxes."""
    box1 = torch.as_tensor(box1).unsqueeze(0)  # Shape: [1, 4]
    box2 = torch.as_tensor(box2).unsqueeze(0)  # Shape: [1, 4]
    return box_iou(box1, box2)[0, 0]


# Genera un file json in cui vengono salvati sia gli embeddings che le immagini delle RoI, è molto pesante.
# Se si tolgono i 3 commenti salva anche le immagini delle RoI, ma è molto pesante.
def generate_class_embeddings(
    feature_extractor, class_list, test_loader, show_plots=False
):
    """Generates embeddings for each class using ResNet50."""
    class_embeddings = {}

    for i, class_name in enumerate(class_list):
        print(f"Generating embedding for class {class_name} (ID: {i})")
        class_embeddings[str(i)] = {
            "name": class_name,
            "label": i,
            "embedding": None,
            # "roi_image": None,
        }

        for images, targets in test_loader:
            for img, target in zip(images, targets):
                class_mask = target["labels"] == i
                if class_mask.any():
                    box = target["boxes"][class_mask][0]
                    img_tensor = img.to(DEVICE)

                    x1, y1, x2, y2 = map(int, box.tolist())
                    roi = img_tensor[:, y1:y2, x1:x2]

                    if show_plots:
                        plot_roi(roi.unsqueeze(0))

                    # roi_numpy = roi.cpu().numpy().tolist()
                    # class_embeddings[str(i)]["roi_image"] = roi_numpy

                    roi = torch.nn.functional.interpolate(
                        roi.unsqueeze(0),
                        size=(362, 362),
                        mode="bilinear",
                        align_corners=False,
                    )

                    roi = normalize_transform(roi)

                    with torch.no_grad():
                        embedding = feature_extractor.forward_one(roi)
                        embedding = embedding.squeeze()
                        class_embeddings[str(i)]["embedding"] = (
                            embedding.cpu().numpy().tolist()
                        )
                    break
            if class_embeddings[str(i)]["embedding"] is not None:
                break

    with open("class_embeddings.json", "w") as f:
        json.dump(class_embeddings, f, indent=4)

    return class_embeddings


def detect_and_classify(
    detector,
    feature_extractor,
    images,
    class_embeddings,
    threshold=0.8,
    show_plots=False,
):
    """Detects and classifies objects in images using embeddings."""
    detector.eval()

    with torch.no_grad():
        predictions = detector([img.to(DEVICE) for img in images])

    batch_results = []
    for image, preds in zip(images, predictions):
        keep = preds["scores"] > threshold
        boxes = preds["boxes"][keep]

        image_results = []
        if len(boxes) > 0:

            if show_plots:
                plot_roi(image)

            rois = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.cpu().tolist())
                roi = image[:, y1:y2, x1:x2]
                roi = torch.nn.functional.interpolate(
                    roi.unsqueeze(0),
                    size=(362, 362),
                    mode="bilinear",
                    align_corners=False,
                )

                roi = normalize_transform(roi)

                rois.append(roi)

            rois_batch = torch.cat(rois, dim=0).to(DEVICE)

            if show_plots:
                plot_roi(roi)

            with torch.no_grad():
                roi_embeddings = feature_extractor.forward_one(rois_batch)
                roi_embeddings = roi_embeddings.squeeze()

            for box_idx, (box, roi_embedding) in enumerate(zip(boxes, roi_embeddings)):
                min_distance = float("inf")
                predicted_class = None

                for class_id, class_data in class_embeddings.items():
                    if class_data["embedding"] is None:
                        continue
                    class_emb = torch.tensor(class_data["embedding"]).to(DEVICE)

                    distance = torch.norm(roi_embedding - class_emb, dim=0).item()

                    if distance < min_distance:
                        min_distance = distance
                        predicted_class = class_data["name"]

                image_results.append(
                    {
                        "box": box.cpu().tolist(),
                        "class": predicted_class,
                        "distance": min_distance,
                    }
                )

        batch_results.append(image_results)

    return batch_results


if __name__ == "__main__":

    class_embeddings = None
    if not os.path.exists("./class_embeddings.json"):
        class_embeddings = generate_class_embeddings(
            feature_extractor, class_list, test_loader, show_plots=False
        )
    else:
        with open("./class_embeddings.json", "r") as f:
            class_embeddings = json.load(f)

    total_images = 0
    correct_images = 0
    total_items = 0
    correct_items = 0
    iou_threshold = 0.5

    for batch_idx, (images, targets) in enumerate(
        tqdm(test_loader, desc="Valutazione test set")
    ):
    
        batch_results = detect_and_classify(
            detector,
            feature_extractor,
            images,
            class_embeddings,
            show_plots=False,
        )

        batch_correct_images = 0
        batch_total_images = len(images)
        batch_correct_items = 0
        batch_total_items = 0

        for i, (image_results, image_targets) in enumerate(zip(batch_results, targets)):
            total_images += 1
            batch_total_items += len(image_targets["boxes"])
            total_items += len(image_targets["boxes"])

            matched_preds = set()
            matched_targets = set()

            for target_idx, target_box in enumerate(image_targets["boxes"]):
                target_label = image_targets["labels"][target_idx]

                best_iou = 0
                best_pred_idx = None

                for pred_idx, pred in enumerate(image_results):
                    if pred_idx in matched_preds:
                        continue

                    iou = calculate_iou(target_box.tolist(), pred["box"])
                    if iou > iou_threshold:
                        if best_iou < iou:
                            best_iou = iou
                            best_pred_idx = pred_idx

                if best_pred_idx is not None:
                    pred_class = image_results[best_pred_idx]["class"]
                    target_class = class_list[target_label]

                    if pred_class == target_class:
                        matched_preds.add(best_pred_idx)
                        matched_targets.add(target_idx)
                        correct_items += 1
                        batch_correct_items += 1

                        if PLOT_COMPARISONS:
                            pred_box = image_results[best_pred_idx]["box"]
                            x1, y1, x2, y2 = map(int, pred_box)
                            pred_roi = images[i][:, y1:y2, x1:x2]

                            target_label_str = str(target_label.item())
                            gt_class_roi = class_embeddings[target_label_str][
                                "roi_image"
                            ]

                            plot_comparison(
                                pred_roi, gt_class_roi, pred_class, target_class
                            )

            if len(matched_preds) == len(image_results) and len(matched_targets) == len(
                image_targets["boxes"]
            ):
                correct_images += 1
                batch_correct_images += 1
                if SHOW_PLOTS:
                    print(
                        f"Numero di elementi nell'immagine: {len(image_targets['boxes'])}"
                    )
                    plot_roi(images[i])

        batch_img_accuracy = batch_correct_images / batch_total_images
        batch_item_accuracy = (
            batch_correct_items / batch_total_items if batch_total_items > 0 else 0
        )
        print(f"\nBatch {batch_idx + 1} Statistics:")
        print(f"Batch Image Accuracy: {batch_img_accuracy:.2%}")
        print(f"Batch Item Accuracy: {batch_item_accuracy:.2%}")
        print(f"Batch Correct Images: {batch_correct_images}/{batch_total_images}")
        print(f"Batch Correct Items: {batch_correct_items}/{batch_total_items}")

    accuracy = correct_images / total_images
    item_accuracy = correct_items / total_items
    print(f"\nAccuratezza Immagini: {accuracy:.2%}")
    print(f"Immagini Corrette: {correct_images}")
    print(f"Immagini Totali: {total_images}")
    print(f"\nAccuratezza Oggetti: {item_accuracy:.2%}")
    print(f"Oggetti Corretti: {correct_items}")
    print(f"Oggetti Totali: {total_items}")

    if os.path.exists("./class_embeddings.json"):

        os.remove("./class_embeddings.json")
