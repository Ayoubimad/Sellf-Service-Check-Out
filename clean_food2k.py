import os
from PIL import Image, UnidentifiedImageError
import logging

def clean_invalid_images(dataset_path):
    logging.basicConfig(filename='invalid_images.log', level=logging.INFO)
    
    removed_count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, filename)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except (IOError, SyntaxError, UnidentifiedImageError) as e:
                    logging.info(f"Removing invalid image: {file_path}")
                    logging.info(f"Error: {str(e)}")
                    os.remove(file_path)
                    removed_count += 1
    
    print(f"Cleaned dataset: removed {removed_count} invalid images")
    print(f"Check 'invalid_images.log' for details of removed files")


def read_image_list(filename):
    images = set()
    try:
        with open(filename, 'r') as f:
            for line in f:
                image_path = line.strip().split()[0]
                images.add(image_path)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return set()
    return images

def filter_and_save(input_file, output_file, reference_images):
    """Filter lines from input file based on reference images and save to output."""
    try:
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in fin:
                image_path = line.strip().split()[0]
                if image_path in reference_images:
                    fout.write(line)
    except FileNotFoundError:
        print(f"Error: Could not process {input_file}")
    

if __name__ == '__main__':

    dataset_path = "/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_split_dataset/"
    # clean_invalid_images(dataset_path)

    food2k_images = set()
    for suffix in ['test', 'train', 'val']:
        images = read_image_list(f'./datasets/Food2k_{suffix}.txt')
        food2k_images.update(images)

    for suffix in ['test', 'train', 'val']:
        filter_and_save(
            f'./datasets/{suffix}_finetune.txt',
            f'./datasets/{suffix}_finetune_filtered.txt',
            food2k_images
        )

    import os
    for suffix in ['test', 'train', 'val']:
        filtered_file = f'./datasets/{suffix}_finetune_filtered.txt'
        original_file = f'./datasets/{suffix}_finetune.txt'
        if os.path.exists(filtered_file):
            os.replace(filtered_file, original_file)
