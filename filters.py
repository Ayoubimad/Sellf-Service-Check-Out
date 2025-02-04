import os

import cv2

"""

NON ESEGUIRE QUESTO SCRIPT, E' GIA' STATO ESEGUITO.

Le immagini filtrate sono in: 

- ./datasets/food-101/filtred_images/
- ./datasets/UNIBIM2016/filtred_original/

"""

output_dir = './datasets/food-101/filtred_images/'
input_dir = './datasets/food-101/images/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dir in os.listdir(input_dir):
    input_subdir = os.path.join(input_dir, dir)

    if not os.path.isdir(input_subdir):
        continue

    output_subdir = os.path.join(output_dir, dir)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for image in os.scandir(input_subdir):
        if image.is_file() and image.name.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(image.path)

            filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

            output_image_path = os.path.join(output_subdir, image.name)
            cv2.imwrite(output_image_path, filtered_img)

output_dir = './datasets/UNIBIM2016/filtred_original/'
input_dir = './datasets/UNIBIM2016/original/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for image in os.scandir(input_dir):
    if image.is_file() and image.name.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(image.path)

        filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        output_image_path = os.path.join(output_dir, image.name)
        cv2.imwrite(output_image_path, filtered_img)


output_dir = './datasets/Food2k_complete_filtered/'
input_dir = './datasets/Food2k_complete/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dir in os.listdir(input_dir):
    input_subdir = os.path.join(input_dir, dir)

    if not os.path.isdir(input_subdir):
        continue

    output_subdir = os.path.join(output_dir, dir)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for image in os.scandir(input_subdir):
        if image.is_file() and image.name.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(image.path)

            filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

            output_image_path = os.path.join(output_subdir, image.name)
            cv2.imwrite(output_image_path, filtered_img)