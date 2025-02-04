from pathlib import Path

import torch
import torch.optim as optim

from faster_rcnn import Faster_RCNN


class FasterRCNNTrainer:

    def __init__(self, config, model=None):

        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = (
            model
            if model is not None
            else Faster_RCNN(num_classes=config.num_classes).get_model()
        )

        self.model.to(self.device)

        self.optimizer = optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        self.start_epoch = 0
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

        self.load_checkpoint()

    def load_checkpoint(self):

        if Path(self.config.checkpoint_path).exists():

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            checkpoint = torch.load(
                self.config.checkpoint_path, map_location=self.device, weights_only=True
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.start_epoch = checkpoint["epoch"]

            self.best_loss = checkpoint.get("best_loss", self.best_loss)

            print(f"Checkpoint loaded. Starting from epoch {self.start_epoch}")

    def train(self, train_loader, valid_loader=None):

        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, targets in train_loader:

                self.optimizer.zero_grad()

                images = [img.to(self.device) for img in images]

                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                losses.backward()

                self.optimizer.step()

                running_loss += losses.item()

            epoch_loss = running_loss / len(train_loader)

            log_message = f"Epoch {epoch + 1}/{self.config.num_epochs}, Training Loss: {epoch_loss:.4f}"

            self.log_training(log_message)
            self.save_checkpoint(epoch, epoch_loss, self.config.checkpoint_path)

            if valid_loader:
                validation_loss = self.validate(valid_loader)
                log_message = f"Epoch {epoch + 1}/{self.config.num_epochs}, Validation Loss: {validation_loss:.4f}"
                self.log_training(log_message)

                if validation_loss < self.best_loss:
                    self.best_loss = validation_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(
                        epoch,
                        validation_loss,
                        self.config.checkpoint_path_best_validation,
                    )
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.config.patience:
                    log_message = f"Early stopping triggered. No improvement for {self.config.patience} epochs."
                    self.log_training(log_message)
                    break

        self.start_epoch = 0

    def validate(self, valid_loader):
        self.model.train()
        running_loss = 0.0
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_loss += losses.item()

        return running_loss / len(valid_loader)

    def log_training(self, message):
        with open(self.config.log_file_path, "a") as log_file:
            log_file.write(message + "\n")

    def save_checkpoint(self, epoch, loss, path):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "best_loss": self.best_loss,
            },
            path,
        )
