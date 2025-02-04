from pathlib import Path
import torch
import os


class Config:

    # directory containing the config file
    BASE_DIR = str(Path(__file__).parent)

    # Base directories
    DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

    # Food-101 dataset paths
    FOOD101_DIR = os.path.join(DATASETS_DIR, "food-101")
    json_train_file = os.path.join(FOOD101_DIR, "meta", "food101_train_new_annotations.json")
    json_test_file = os.path.join(FOOD101_DIR, "meta", "food101_test_new_annotations.json")
    json_valid_file = os.path.join(FOOD101_DIR, "meta", "food101_valid_new_annotations.json")
    image_dir = os.path.join(FOOD101_DIR, "filtred_images")
    roi_dataset_path = os.path.join(FOOD101_DIR, "split_dataset", "train")

    # UNIMIB dataset paths
    UNIMIB_DIR = os.path.join(DATASETS_DIR, "UNIBIM2016")
    unimib_json_file = os.path.join(UNIMIB_DIR, "UNIMIB2016-annotations", "annotations.json")
    unimib_image_dir = os.path.join(UNIMIB_DIR, "original")
    unimib_mat_file = os.path.join(UNIMIB_DIR, "split", "original_food_list.mat")

    # Food2k dataset paths
    FOOD2K_DIR = os.path.join(DATASETS_DIR, "Food2k_split_dataset")
    food2k_dataset_path_train = os.path.join(FOOD2K_DIR, "train")
    food2k_dataset_path_test = os.path.join(FOOD2K_DIR, "test")
    food2k_dataset_path_valid = os.path.join(FOOD2K_DIR, "val")
    food2k_train_txt = os.path.join(DATASETS_DIR, "Food2k_train.txt")
    food2k_val_txt = os.path.join(DATASETS_DIR, "Food2k_val.txt")

    # Checkpoint and model paths
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "food_detector.pt")
    checkpoint_path_best_validation = os.path.join(CHECKPOINTS_DIR, "best_food_detector.pt")
    final_model_path = os.path.join(WEIGHTS_DIR, "faster_rcnn.pth")
    siamese_model_path = os.path.join(CHECKPOINTS_DIR, "siamese_checkpoint.pth")
    siamese_checkpoint_dir = CHECKPOINTS_DIR

    # Log paths
    log_file_path = os.path.join(LOGS_DIR, "food_detector_training.logs")
    siamese_log_filename = os.path.join(LOGS_DIR, "training_siamese.log")

    # Faster R-CNN configurations
    batch_size = 4
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    num_epochs = 100
    encoder_classes = ["food"]
    num_classes = 2  # Food + Background
    patience = 10

    # Siamese Network configurations
    siamese_epochs = 100
    siamese_learning_rate = 1e-4
    embedding_dim = 128
    margin = 0.85
    mining_margin = 0.85
    mining_chunk_size = 16
    triplet_batch_size = 16
    embedding_chunk_size = 32
    siamese_batch_size = 512 # più alto è meglio è. Da 512 generiamo le triplette. Se basso, non si hanno abbastanza triplette per fare il mining
    siamese_weight_decay = 0.01

    # Mining parameters
    initial_mining_type = "semihard"
    later_mining_type = "semihard"
    mining_switch_epoch = 999999 # never switch

    # Dataset parameters
    n_samples_per_class = 5
    num_workers = 4

    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_frequency = 1
    resume_training = True

    # Early stopping parameters
    early_stopping_patience = 10
    min_delta = 0.001
