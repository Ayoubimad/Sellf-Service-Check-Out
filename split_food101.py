import os
import shutil
import random

def split_dataset_by_class(data_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Suddivide un dataset organizzato in cartelle per classe in train e test.

    Args:
        data_dir (str): Path al dataset originale.
        output_dir (str): Path alla directory dove salvare il dataset suddiviso.
        train_ratio (float): Percentuale di immagini da utilizzare per il training.
        seed (int): Seme casuale per la riproducibilità.
    """
    random.seed(seed)

    # Crea le directory di output per train e test
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Per ogni classe nel dataset
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        # Controlla se è una directory
        if not os.path.isdir(class_path):
            continue

        # Crea le directory di output per questa classe
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Ottieni tutte le immagini della classe
        images = os.listdir(class_path)
        random.shuffle(images)

        # Determina il numero di immagini per il train
        train_size = int(len(images) * train_ratio)

        # Dividi le immagini in train e test
        train_images = images[:train_size]
        test_images = images[train_size:]

        # Sposta le immagini nelle directory appropriate
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

        print(f"Classe '{class_name}' suddivisa: {len(train_images)} train, {len(test_images)} test.")

# Path al dataset originale e alla destinazione
original_dataset_dir = "/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_complete"
output_dataset_dir = "/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_complete/split_dataset"

# Suddivisione
split_dataset_by_class(original_dataset_dir, output_dataset_dir, train_ratio=0.8)
