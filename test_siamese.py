# adattato da train_embeddings.py
import os
import random

import torch
import PIL
from torch.utils.data import DataLoader
import logging
import json
from datetime import datetime
import pathlib

from torchvision import transforms

from siamese import SiameseNetwork

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

# dataset.ImageFolder carica tutto il dataset in RAM, succede che si va in OOM
# la classe carica solo i dati necessari quando serve

# da muovere in datasets
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, image_path, transform=None, augment=None, loader=My_loader, return_triplets=False):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1].strip())))
        self.imgs = imgs
        self.transform = transform
        self.augment = augment
        self.loader = loader
        self.image_path = image_path
        self.return_triplets = return_triplets
        
        self.images_by_class = {}
        for img_name, label in self.imgs:
            if label not in self.images_by_class:
                self.images_by_class[label] = []
            self.images_by_class[label].append(img_name)

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        img = self.loader(os.path.join(self.image_path, img_name))

        if not self.return_triplets:
            if self.transform is not None:
                img = self.transform(img)
            if self.augment is not None:
                img = self.augment(img)
            return img, label
        
        pos_img_name = random.choice(self.images_by_class[label])
        while pos_img_name == img_name:
            pos_img_name = random.choice(self.images_by_class[label])
        pos_img = self.loader(os.path.join(self.image_path, pos_img_name))
        
        neg_label = random.choice(list(set(self.images_by_class.keys()) - {label}))
        neg_img_name = random.choice(self.images_by_class[neg_label])
        neg_img = self.loader(os.path.join(self.image_path, neg_img_name))

        if self.transform is not None:
            img = self.transform(img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        if self.augment is not None:
            img = self.augment(img)
            pos_img = self.augment(pos_img)
            neg_img = self.augment(neg_img)

        return img, pos_img, neg_img

def calculate_metrics(model, dataloader, device):
    logging.info("Starting metrics calculation...")
    model.eval()
    correct = 0
    total = 0
    all_pos_distances = []
    all_neg_distances = []

    with torch.no_grad():
        
        for anchor, positive, negative in dataloader:

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            if isinstance(model, torch.nn.DataParallel):
                model = model.module

            anchor_out = model.forward_one(anchor)
            positive_out = model.forward_one(positive)
            negative_out = model.forward_one(negative)

            pos_dist = torch.norm(anchor_out - positive_out, dim=1)
            neg_dist = torch.norm(anchor_out - negative_out, dim=1)

            correct += torch.sum(pos_dist < neg_dist).item()
            total += anchor.size(0)

            all_pos_distances.extend(pos_dist.cpu().numpy())
            all_neg_distances.extend(neg_dist.cpu().numpy())

    accuracy = correct / total
    
    mean_pos_dist = torch.mean(torch.tensor(all_pos_distances)).item()
    mean_neg_dist = torch.mean(torch.tensor(all_neg_distances)).item()
    
    metrics = {
        'accuracy': accuracy,
        'mean_positive_distance': mean_pos_dist,
        'mean_negative_distance': mean_neg_dist,
    }
    
    logging.info(f"Metrics calculated: {json.dumps(metrics, indent=2)}")
    return metrics

if __name__ == "__main__":

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"test_siamese_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )

    split_data_dir = "/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_complete"

    data_transforms = transforms.Compose([
        transforms.Resize((362, 362)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = MyDataset(
        txt_dir='/work/cvcs2024/SelfService_CheckOut/datasets/test_finetune.txt',
        image_path='/work/cvcs2024/SelfService_CheckOut/datasets/Food2k_complete',
        transform=data_transforms,
        return_triplets=True
    )

    batch_size = 128 * torch.cuda.device_count() if torch.cuda.device_count() > 1 else 128

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    checkpoint_dir = pathlib.Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    model = SiameseNetwork()
    model_path = "./weights/siamese.pth"
    logging.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    metrics_list = []
    last_checkpoint_path = checkpoint_dir / f"last_checkpoint.pth"
    
    start_iteration = 0
    if last_checkpoint_path.exists():
        logging.info(f"Loading last checkpoint from {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path)
        start_iteration = checkpoint['iteration']
        metrics_list = checkpoint.get('metrics_list', [])
        
    for i in range(start_iteration, 10):
        logging.info(f"\nStarting iteration {i+1}/10")
        metrics = calculate_metrics(model, test_loader, device)
        metrics_list.append(metrics)
        
        checkpoint = {
            'iteration': i+1,
            'metrics': metrics,
            'metrics_list': metrics_list,
        }
        
        torch.save(checkpoint, last_checkpoint_path)
        logging.info(f"Saved checkpoint to {last_checkpoint_path}")

    mean_metrics = {}
    std_metrics = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        mean_metrics[metric] = sum(values) / len(values)
        std_metrics[metric] = torch.std(torch.tensor(values)).item()

    final_results = {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics
    }
    
    results_path = log_dir / f"final_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logging.info("\nFinal Results:")
    for metric in mean_metrics.keys():
        logging.info(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
