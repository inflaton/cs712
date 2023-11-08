import os
import random
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
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
def checkpoint_load(model, save_path, epoch, n_classes=0):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    model.load_state_dict(torch.load(f))
    print("loaded checkpoint:", f, flush=True)


def load_training_data(epoch, max_fold=10):
    # Load the labels
    labels = np.loadtxt(f"data/train/label_train.txt")
    labels = torch.from_numpy(labels).long()

    data = None
    filename = (
        "data/timm_preprocessed_train.npy"
        if max_fold == 0
        else f"data/timm_preprocessed_train_{epoch%max_fold}.npy"
    )
    preprocessed_data = np.load(filename)
    # Convert the NumPy array to PyTorch tensors
    data = torch.from_numpy(preprocessed_data).float()
    print(f"loaded training data from: {filename}")
    return data, labels


def train_model(
    model,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    max_fold,
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
            scheduler.step()
            continue

        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
        print(f"Epoch {epoch + 1}, Learning rate: {learning_rate:.10f}", flush=True)

        data, labels = load_training_data(epoch, max_fold=max_fold)

        # Define the dataset and dataloader
        dataset = JigsawDataset(data, labels)
        train_set, val_set = random_split(dataset, [0.9, 0.1])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

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

        scheduler.step()

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


if __name__ == "__main__":
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=64)
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        help="Number of folds of data augmentation",
        default=10,
    )
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
    max_fold = args.fold

    print("epochs: ", num_epochs, "batch:", batch_size, "max_fold:", max_fold)

    # Create the model
    # model = JigsawModel(n_classes=num_classes).to(device)
    model = JigsawNet(
        n_classes=num_classes, num_features=3072, relu_in_last_fc=True
    ).to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(
        model,
        optimizer,
        scheduler,
        criterion,
        num_epochs,
        max_fold,
        load_checkpoint,
    )
