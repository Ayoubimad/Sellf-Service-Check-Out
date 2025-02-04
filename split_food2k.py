import os
import shutil
import random

def split_dataset_by_class(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=42):

    random.seed(seed)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    count = 0

    class_list = sorted(os.listdir(data_dir))

    for class_name in class_list:
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            print(f"Class '{class_name}' is not a directory")
            continue

        # nel dataset di Food2k esiste una cartella chiamata 'split_dataset' da non usare
        if os.path.dirname(class_path) == 'split_dataset':
            continue
    
        count += 1
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        
        for dir_path in [train_class_dir, val_class_dir, test_class_dir]:
            os.makedirs(dir_path, exist_ok=True)


        images = os.listdir(class_path)
        random.shuffle(images)

        total_images = len(images)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)

        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        for img, target_dir in [
            (train_images, train_class_dir),
            (val_images, val_class_dir),
            (test_images, test_class_dir)
        ]:
            for image in img:
                shutil.copy(
                    os.path.join(class_path, image),
                    os.path.join(target_dir, image)
                )

    print(f"Total classes: {count}")

if __name__ == "__main__":
    data_dir = "/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_complete"
    output_dir = "/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_split_dataset"

    split_dataset_by_class(data_dir, output_dir)

    print(f"Number of classes in train: {len(os.listdir(os.path.join(output_dir, 'train')))}")
    print(f"Number of classes in val: {len(os.listdir(os.path.join(output_dir, 'val')))}")
    print(f"Number of classes in test: {len(os.listdir(os.path.join(output_dir, 'test')))}")
