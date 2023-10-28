import os
import random
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from JigsawNet import JigsawNet


class JigsawDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        label = self.labels[idx]
        return puzzle, label


# save checkpoint function
def checkpoint_save(model, save_path, epoch):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    torch.save(model.state_dict(), f)
    print("saved checkpoint:", f, flush=True)


# save checkpoint function
def checkpoint_load(model, save_path, epoch, n_classes=0, model_ver=1):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    model.load_state_dict(torch.load(f))
    print("loaded checkpoint:", f, flush=True)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    load_checkpoint=False,
):
    save_path = os.path.join(os.getcwd(), "data", "checkpoints/")
    os.makedirs(save_path, exist_ok=True)

    loaded_checkpoint = -1
    if load_checkpoint:
        for _, _, files in os.walk(save_path):
            for filename in files:
                checkpoint = int(re.split("[-.]", filename)[-2])
                if checkpoint > loaded_checkpoint:
                    loaded_checkpoint = checkpoint

        checkpoint_load(model, save_path, loaded_checkpoint)

    highest_accuracy = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        if epoch <= loaded_checkpoint:
            continue

        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        print(f"Epoch {epoch + 1}, Learning rate: {learning_rate:.10f}", flush=True)

        for puzzle, label in train_loader:
            puzzle, label = puzzle.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(puzzle)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted == label).sum().item()

        # scheduler.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%",
            flush=True,
        )

        # Validation for this epoch
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            val_loss = []
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                batch_loss_value = loss.item()
                val_loss.append(batch_loss_value)

                # Calculate accuracy for this batch
                _, predicted = torch.max(outputs, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

            # Print statistics
            loss_value = np.mean(val_loss)
            accuracy = correct_predictions / total_predictions
            print(
                f"Epoch {epoch + 1}, Validation Loss: {loss_value:.4f}, Validation Accuracy: {accuracy * 100:.2f}%",
                flush=True,
            )

            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_epoch = epoch

            checkpoint_save(model, save_path, epoch)

    print(
        f"Best epoch {best_epoch + 1}, Highest Validation Accuracy: {highest_accuracy * 100:.2f}%",
        flush=True,
    )


RANDOM_SEED = 193
# RANDOM_SEED = 194
# RANDOM_SEED = 1940

# initialising seed for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
seeded_generator = torch.Generator().manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")  # Use CPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=64)
    parser.add_argument(
        "-r",
        "--resume",
        type=int,
        help="Resume from previous training checkpoint",
        default=0,
    )
    parser.add_argument(
        "-l",
        "--learing_rate",
        type=float,
        help="Learning rate",
        default=0.001,
    )

    # Parse the arguments
    args = parser.parse_args()
    print(f"device: {device}")

    num_classes = 50
    batch_size = args.batch
    num_epochs = args.epochs
    load_checkpoint = args.resume > 0
    learing_rate = args.learing_rate

    print("epochs: ", num_epochs, "batch", batch_size)

    # Create the model
    # model = JigsawModel(n_classes=num_classes).to(device)
    model = JigsawNet(n_classes=num_classes).to(device)

    # Define the dataset and dataloader
    dataset = JigsawDataset(data, labels)
    print(f"dataset len: {len(dataset)}")

    train_set, val_set = random_split(dataset, [0.8, 0.2])
    print(f"train_set len: {len(train_set)}")
    print(f"val_set len: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)
    # optimizer = optim.AdamW(
    # model.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.85, 0.999)
    # )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        num_epochs,
        load_checkpoint,
    )
