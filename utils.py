import os
import json
import random
from typing import Tuple, List, Optional

import numpy as np
import scipy.io
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import supervision as sv

from transform import Transform


def remove_0_(folder_path: str) -> None:
    """
    Rimuove la sottostringa "(0)" dai nomi dei file immagine nella cartella specificata, se presente.

    Questa funzione è utile quando alcune immagini in una cartella sono state salvate con "(0)" nei loro nomi
    (es. "__image_name_(0)_jpg"), a causa di un problema con il processo di decompressione.

    Args:
        folder_path (str): Percorso della cartella contenente le immagini.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") and "(0)" in filename:
            new_name = filename.replace("(0)", "").strip()
            original_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)
            os.rename(original_file, new_file)
            print(f"File rinominato: {filename} -> {new_name}")


def load_image(image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Carica un'immagine da un percorso specificato e applica le trasformazioni.

    Args:
        image_path (str): Percorso dell'immagine.

    Returns:
        Tuple[np.array, torch.Tensor]: Restituisce l'immagine in due formati:
            - np.array: Immagine originale come ndarray.
            - torch.Tensor: Tensore dell'immagine trasformata con valori normalizzati tra [0,1].
    """
    transform = Transform()
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed


def get_random_image_path(directory: str) -> str:
    """
    Seleziona un'immagine casuale da una directory specificata.

    Args:
        directory (str): Percorso della directory contenente le immagini.

    Returns:
        str: Percorso completo dell'immagine selezionata casualmente.
    """
    images = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg")
    ]
    return random.choice(images)


def annotate(
    image_source: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    class_id: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    phrases: Optional[List[str]] = None,
    denormalize: bool = False,
    bbox_thickness=5,
    text_scale=4,
    text_color=sv.Color.BLACK,
) -> np.ndarray:
    """
    Annota un'immagine.

    Riferimento: @ https://supervision.roboflow.com/latest/

    Args:
        image_source: L'immagine sorgente con forma (HxWxC).
        boxes: Array di forma (n, 4) contenente le coordinate dei box in formato [x1, y1, x2, y2].
        masks: Array di forma (n, H, W) contenente le maschere di segmentazione.
        class_id: Array di forma (n,) contenente gli ID delle classi delle rilevazioni.
        confidence: Array di forma (n,) contenente i punteggi di confidenza delle rilevazioni.
        phrases: Etichette per ogni bounding box.
        denormalize: Se True, denormalizza i box in base alle dimensioni dell'immagine.

    Returns:
        np.ndarray: Immagine annotata con bounding box ed etichette.
    """

    h, w, _ = image_source.shape

    if denormalize:
        boxes = boxes * torch.Tensor([w, h, w, h])

    detections = sv.Detections(
        xyxy=boxes, mask=masks, confidence=confidence, class_id=class_id
    )

    bbox_annotator = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.CLASS, thickness=bbox_thickness
    )

    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.CLASS,
        text_scale=text_scale,
        text_color=text_color,
    )

    mask_annotator = sv.MaskAnnotator()

    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    if boxes is not None:
        annotated_frame = bbox_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

    if masks is not None:
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

    if phrases is not None:
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=phrases,
        )

    return annotated_frame


def denormalize_boxes(
    boxes: torch.Tensor, img_width: int, img_height: int
) -> torch.Tensor:
    """
    Denormalizza le coordinate dei bounding box alle dimensioni originali dell'immagine.

    Args:
        boxes (torch.Tensor): Bounding box normalizzati.
        img_width (int): Larghezza dell'immagine.
        img_height (int): Altezza dell'immagine.

    Returns:
        torch.Tensor: Bounding box denormalizzati.
    """
    w, h = img_width, img_height
    return boxes * torch.Tensor([w, h, w, h])


def normalize_boxes(
    boxes: torch.Tensor, img_width: int, img_height: int
) -> torch.Tensor:
    """
    Normalizza le coordinate dei bounding box a valori tra [0,1].

    Args:
        boxes (torch.Tensor): Bounding box nelle dimensioni originali dell'immagine.
        img_width (int): Larghezza dell'immagine.
        img_height (int): Altezza dell'immagine.

    Returns:
        torch.Tensor: Bounding box normalizzati.
    """
    w, h = img_width, img_height
    return boxes / torch.Tensor([w, h, w, h])


def plot_tsne(features, labels, title, save_path):
    """
    Visualizza t-SNE delle features e salva su file

    Args:
        features: Embedding delle features (n_samples, n_features)
        labels: Etichette delle classi per ogni campione
        title: Titolo del grafico
        save_path: Percorso per salvare l'immagine del grafico
    """
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.detach().cpu().numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels.cpu().numpy(), cmap="tab10"
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    os.makedirs("./tsne_plots", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def clean_annotations(UNIMIB_JSON_FILE, UNIMIB_MAT_FILE, OUTPUT_JSON_FILE):
    class_list = scipy.io.loadmat(UNIMIB_MAT_FILE)["final_food_list"]
    valid_classes = set(str(item[0][0]) for item in class_list)

    with open(UNIMIB_JSON_FILE, "r") as f:
        data = json.load(f)

    cleaned_annotations = []
    removed_count = 0

    for img_ann in data["annotations"]:
        image_name = img_ann["image_name"]
        valid_annotations = []

        for ann in img_ann["annotations"]:
            if "class" not in ann:
                removed_count += 1
                print(
                    f"Rimozione annotazione senza informazioni sulla classe da {image_name}"
                )
                continue

            if ann["class"] in valid_classes:
                valid_annotations.append(ann)
            else:
                removed_count += 1
                print(f"Rimozione classe non valida: {ann['class']} da {image_name}")

        if valid_annotations:
            cleaned_annotations.append(
                {"image_name": image_name, "annotations": valid_annotations}
            )

    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump({"annotations": cleaned_annotations}, f, indent=4)

    print(f"\nPulizia completata:")
    print(f"- Rimosse {removed_count} annotazioni non valide")
    print(f"- Immagini originali: {len(data['annotations'])}")
    print(f"- Immagini pulite: {len(cleaned_annotations)}")
    print(f"\nAnnotazioni pulite salvate in: {OUTPUT_JSON_FILE}")
