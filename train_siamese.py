import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_metric_learning.miners import TripletMarginMiner
from torchvision import transforms
from config import Config
from siamese import SiameseNetwork
from test_siamese import MyDataset
from transform import Augment

class SiameseTrainer:
    def __init__(self, config: Config):
        self.config = config
        
        self.patience = self.config.early_stopping_patience
        self.min_delta = self.config.min_delta
        self.best_val_loss = float('inf')
        self.counter = 0

        self.augment = Augment()

        self.data_transforms = transforms.Compose([
        transforms.Resize((362, 362)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._setup_logging()
        self._setup_data()
        self._setup_model()

    def _setup_logging(self):
        logging.basicConfig(
            filename=self.config.siamese_log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    def _setup_data(self):

        train_dataset = MyDataset(
            txt_dir=self.config.food2k_train_txt,
            image_path=self.config.food2k_dataset_path_train,
            transform=self.data_transforms,
            augment=self.augment,
            return_triplets=False, # genero triplette dopo con TripletMarginMiner
        )
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.siamese_batch_size, shuffle=True)

        val_dataset = MyDataset(
            txt_dir=self.config.food2k_val_txt,
            image_path=self.config.food2k_dataset_path_valid,
            transform=self.data_transforms,
            return_triplets=True,
        )
        self.val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    def _setup_model(self):
        self.model = SiameseNetwork().to(self.config.device)
        logging.info(f"Model created and moved to device: {self.config.device}")
        
        gpu_count = torch.cuda.device_count()
        logging.info(f"Number of available GPUs: {gpu_count}")
        
        if gpu_count > 1:
            logging.info(f"Using {gpu_count} GPUs")
            self.model = nn.DataParallel(self.model)
            logging.info("Model wrapped with DataParallel")
        else:
            logging.info("Running on single GPU/CPU")
        
        self.optimizer = self._get_optimizer()
        logging.info(f"Optimizer initialized with learning rate: {self.config.siamese_learning_rate}")
        
        self.triplet_loss = nn.TripletMarginLoss(margin=self.config.margin)
        logging.info(f"TripletMarginLoss initialized with margin: {self.config.margin}")

    def _get_optimizer(self):
        optimizer = optim.AdamW(
            [
                {
                    "params": self.model.parameters(),
                    "lr": self.config.siamese_learning_rate,
                }
            ],
            weight_decay=self.config.siamese_weight_decay,
        )
        logging.info(f"AdamW optimizer created with weight_decay: {self.config.siamese_weight_decay}")
        return optimizer

    def get_embeddings_in_chunks(self, images):
        embeddings = []
        self.model.eval()

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for i in range(0, len(images), self.config.embedding_chunk_size):
            chunk = images[i : i + self.config.embedding_chunk_size]
            chunk = chunk.to(self.config.device)
            with torch.no_grad():
                chunk_embeddings = model.forward_one(chunk)
            embeddings.append(chunk_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def get_triplets(self, features, targets, mining_type, epoch):
        miner = TripletMarginMiner(
            type_of_triplets=mining_type, margin=self.config.mining_margin
        )
        indices_tuple = miner(features.cpu(), targets.cpu())

        if indices_tuple[0].numel() == 0:
            return torch.tensor([])

        triplets = [(a.item(), p.item(), n.item()) for a, p, n in zip(*indices_tuple)]

        return torch.tensor(triplets)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_triplets = 0
        num_batches = 0

        mining_type = (
            self.config.initial_mining_type
            if epoch < self.config.mining_switch_epoch
            else self.config.later_mining_type
        )

        logging.info(f"Epoch {epoch}: Using {mining_type} mining")

        for batch_idx, (images, targets) in enumerate(self.train_dataloader):
            triplets = self._mine_triplets(images, targets, mining_type, epoch)
            if len(triplets) == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx}: No triplets found, skipping")
                continue

            batch_loss = self._process_triplets(images, triplets)
            

            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx} of {len(self.train_dataloader)}: Loss = {batch_loss:.4f}, Triplets = {len(triplets)}")


            total_loss += batch_loss
            total_triplets += len(triplets)
            num_batches += 1

        return self._compute_epoch_metrics(total_loss, total_triplets, num_batches)

    def _mine_triplets(self, images, targets, mining_type, epoch):
        with torch.no_grad():
            self.model.eval()
            embeddings = self.get_embeddings_in_chunks(images)
            return self.get_triplets(embeddings, targets, mining_type, epoch)

    def _process_triplets(self, images, triplets):
        batch_loss = 0
        n_batches = 0

        for i in range(0, len(triplets), self.config.triplet_batch_size):
            batch_triplets = triplets[i : i + self.config.triplet_batch_size]
            if len(batch_triplets) < 2:
                continue

            loss = self._compute_triplet_loss(images, batch_triplets)
            batch_loss += loss.item()
            n_batches += 1

        return batch_loss / max(1, n_batches)

    def _compute_triplet_loss(self, images, batch_triplets):
        self.optimizer.zero_grad()

        anchor_imgs = images[batch_triplets[:, 0]].to(self.config.device)
        pos_imgs = images[batch_triplets[:, 1]].to(self.config.device)
        neg_imgs = images[batch_triplets[:, 2]].to(self.config.device)

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        anchor_emb, positive_emb, negative_emb = model(
            anchor_imgs, pos_imgs, neg_imgs
        )

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        if torch.isnan(loss):
            logging.error("NaN loss detected")
            return torch.tensor(0.0, device=self.config.device)
            
        loss.backward()
        self.optimizer.step()

        return loss

    def _compute_epoch_metrics(self, total_loss, total_triplets, num_batches):
        avg_loss = total_loss / max(1, num_batches)
        avg_triplets = total_triplets / max(1, num_batches)
        return avg_loss, avg_triplets

    def save_checkpoint(self, epoch, loss):
        os.makedirs(self.config.siamese_checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "best_val_loss": self.best_val_loss
        }
        torch.save(checkpoint, self.config.siamese_model_path)
        logging.info(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(
                self.config.siamese_model_path, map_location=self.config.device
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            loss = checkpoint["loss"]
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            logging.info(f"Resumed from epoch {start_epoch} with loss {loss:.4f} and best validation loss {self.best_val_loss:.4f}")
            return start_epoch
        except FileNotFoundError:
            logging.info("No checkpoint found, starting from scratch")
            return 0

    def validate(self):
        self.model.eval()
        total_val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for anchor_img, pos_img, neg_img in self.val_dataloader:
                
                batch_loss = 0
                n_sub_batches = 0

                        
                anchor_imgs = anchor_img.to(self.config.device)
                pos_imgs = pos_img.to(self.config.device)
                neg_imgs = neg_img.to(self.config.device)

                model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    
                anchor_emb, positive_emb, negative_emb = model(anchor_imgs, pos_imgs, neg_imgs)

                loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
                    
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                
        return total_val_loss / max(1, len(self.val_dataloader))

    def train(self):
        start_epoch = self.load_checkpoint() if self.config.resume_training else 0
        logging.info(f"Starting training from epoch {start_epoch}")
        
        logging.info(
            f"Training configuration: batch_size={self.config.siamese_batch_size}, "
            f"lr={self.config.siamese_learning_rate}, epochs={self.config.siamese_epochs}, "
            f"patience={self.patience}, min_delta={self.min_delta}"
        )

        for epoch in range(start_epoch, self.config.siamese_epochs):
            epoch_start_time = time.time()
            avg_loss, avg_triplets = self.train_epoch(epoch)
            val_loss = self.validate()
            epoch_time = time.time() - epoch_start_time

            logging.info(
                f"Epoch {epoch} complete in {epoch_time:.2f}s. Training Loss: {avg_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, Average Triplets per batch: {avg_triplets:.2f}"
            )

            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.counter = 0
                self.save_checkpoint(epoch, avg_loss)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            if epoch % 1 == 0:
                self.save_checkpoint(epoch, avg_loss)


if __name__ == "__main__":
    config = Config()
    trainer = SiameseTrainer(config)
    trainer.train()
