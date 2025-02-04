import json
import random
from collections import defaultdict

json_file = "./datasets/food-101/meta/food101_new_annotations.json"
with open(json_file, "r") as f:
    data = json.load(f)

class_dict = defaultdict(list)

for item in data:
    class_name = item["image_name"].split("/")[0]
    class_dict[class_name].append(item)

train_data = []
valid_data = []
test_data = []

for class_name, items in class_dict.items():
    random.shuffle(items)
    train_split = int(0.7 * len(items))
    valid_split = int(0.80 * len(items))

    train_data.extend(items[:train_split])
    valid_data.extend(items[train_split:valid_split])
    test_data.extend(items[valid_split:])

random.shuffle(train_data)
random.shuffle(valid_data)
random.shuffle(test_data)

print(len(train_data), len(valid_data), len(test_data))


train_json_file = "./datasets/food-101/meta/food101_train_new_annotations.json"
valid_json_file = "./datasets/food-101/meta/food101_valid_new_annotations.json"
test_json_file = "./datasets/food-101/meta/food101_test_new_annotations.json"

with open(train_json_file, "w") as f:
    json.dump(train_data, f, indent=4)

with open(valid_json_file, "w") as f:
    json.dump(valid_data, f, indent=4)

with open(test_json_file, "w") as f:
    json.dump(test_data, f, indent=4)

train_class_count = defaultdict(int)
valid_class_count = defaultdict(int)
test_class_count = defaultdict(int)

for item in train_data:
    class_name = item["image_name"].split("/")[0]
    train_class_count[class_name] += 1

for item in valid_data:
    class_name = item["image_name"].split("/")[0]
    valid_class_count[class_name] += 1

for item in test_data:
    class_name = item["image_name"].split("/")[0]
    test_class_count[class_name] += 1

print("\nNumero di elementi nel set di training:")
for class_name, count in train_class_count.items():
    print(f"Classe '{class_name}': {count} immagini")

print("\nNumero di elementi nel set di validazione:")
for class_name, count in valid_class_count.items():
    print(f"Classe '{class_name}': {count} immagini")

print("\nNumero di elementi nel set di test:")
for class_name, count in test_class_count.items():
    print(f"Classe '{class_name}': {count} immagini")

################################################################################

train_file = "./datasets/food-101/meta/train.txt"
valid_file = "./datasets/food-101/meta/valid.txt"

with open(train_file, "r") as f:
    lines = f.readlines()

class_images = defaultdict(list)
for line in lines:
    line = line.strip()
    class_name = line.split("/")[0]
    class_images[class_name].append(line)

valid_lines = []
new_train_lines = []

for class_name, images in class_images.items():
    num_valid = int(len(images) * 0.2)
    valid_subset = random.sample(images, num_valid)
    valid_lines.extend(valid_subset)
    new_train_lines.extend([img for img in images if img not in valid_subset])

random.shuffle(valid_lines)
random.shuffle(new_train_lines)

with open(valid_file, "w") as f:
    f.write("\n".join(valid_lines))

with open(train_file, "w") as f:
    f.write("\n".join(new_train_lines))

print(f"Original training set size: {len(lines)}")
print(f"New training set size: {len(new_train_lines)}")
print(f"Validation set size: {len(valid_lines)}")

train_class_count = defaultdict(int)
valid_class_count = defaultdict(int)

for line in new_train_lines:
    class_name = line.split("/")[0]
    train_class_count[class_name] += 1

for line in valid_lines:
    class_name = line.split("/")[0]
    valid_class_count[class_name] += 1

print("\nClass distribution in training set:")
for class_name, count in train_class_count.items():
    print(f"{class_name}: {count}")

print("\nClass distribution in validation set:")
for class_name, count in valid_class_count.items():
    print(f"{class_name}: {count}")
