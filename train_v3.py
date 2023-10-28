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
from sklearn.model_selection import KFold


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


def checkpoint_delete(save_path, epoch):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    os.remove(f)


# save checkpoint function
def checkpoint_load(model, save_path, epoch, n_classes=0, model_ver=1):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    model.load_state_dict(torch.load(f))
    print("loaded checkpoint:", f, flush=True)


def load_training_data():
    # Load the labels
    labels = np.loadtxt(f"data/train/label_train.txt")
    labels = torch.from_numpy(labels).long()

    filename = "data/preprocessed_train.npy"
    path = Path(filename)

    data = None
    if path.is_file():
        preprocessed_data = np.load(filename)
        # Convert the NumPy array to PyTorch tensors
        data = torch.from_numpy(preprocessed_data).float()
        print(f"loaded training data from: {filename}")
    else:
        fold = 0
        labels_one_fold = labels
        while True:
            filename = f"data/preprocessed_train_{fold}.npy"
            path = Path(filename)

            if path.is_file():
                preprocessed_data = np.load(filename)
                # Convert the NumPy array to PyTorch tensors
                temp = torch.from_numpy(preprocessed_data).float()

                if fold > 0:
                    data = ConcatDataset([data, temp])
                    labels = ConcatDataset([labels, labels_one_fold])
                else:
                    data = temp

                print(f"loaded training data from: {filename}")
                fold += 1
            else:
                break

    return data, labels


def get_k_fold_training_datasets(kfold, dataset, fold):
    print(f"get_k_fold_training_datasets: fold={fold}")

    for idx, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        if idx == fold:
            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)
            return train_subset, val_subset


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

    for _, _, files in os.walk(save_path):
        for filename in files:
            checkpoint = int(re.split("[-.]", filename)[-2])
            if load_checkpoint:
                checkpoint_load(model, save_path, checkpoint)
            checkpoint_delete(save_path, checkpoint)

    highest_accuracy = 0
    best_epoch = -1

    for epoch in range(num_epochs):
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
                if best_epoch >= 0:
                    checkpoint_delete(save_path, best_epoch)

                highest_accuracy = accuracy
                best_epoch = epoch

                checkpoint_save(model, save_path, epoch)

    print(
        f"Best epoch {best_epoch + 1}, Highest Validation Accuracy: {highest_accuracy * 100:.2f}%",
        flush=True,
    )


class JigsawValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        return puzzle


def evaluate_model(model, data_loader):
    save_path = os.path.join(os.getcwd(), "data", "checkpoints/")

    for _, _, files in os.walk(save_path):
        filename = files[0]
        checkpoint = int(re.split("[-.]", filename)[-2])
        checkpoint_load(model, save_path, checkpoint)

    model.eval()
    all_predictions = []  # To store translated predictions

    with torch.no_grad():
        for puzzle in data_loader:
            puzzle = puzzle.to(device)
            output = model(puzzle)
            _, predicted = torch.max(
                output, 1
            )  # Get the index of the max log-probability
            all_predictions.extend(predicted.cpu().detach().numpy())

    all_predictions = np.array(all_predictions)
    all_predictions = all_predictions.astype(int)

    # Save the predicted values to a text file
    filename = "data/validation.txt"
    np.savetxt(filename, all_predictions, fmt="%d")

    # compress the results folder
    zip_filename = "data/result.zip"
    path = Path(zip_filename)
    if path.is_file():
        os.remove(zip_filename)
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(filename, arcname="validation.txt")

    print(f"results saved to: {zip_filename}")


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
    parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)

    # Parse the arguments
    args = parser.parse_args()
    print(device)

    num_classes = 50
    batch_size = args.batch
    num_epochs = args.epochs

    print("epochs: ", num_epochs, "batch", batch_size)

    # Create the model
    # model = JigsawModel(n_classes=num_classes).to(device)
    model = JigsawNet(n_classes=num_classes).to(device)

    if num_epochs > 0:
        data, labels = load_training_data()

        kfold = KFold(n_splits=5, shuffle=True)

        # Define the dataset and dataloader
        dataset = JigsawDataset(data, labels)
        print(f"dataset len: {len(dataset)}")

        for fold in range(5):
            train_set, val_set = get_k_fold_training_datasets(kfold, dataset, fold)

            print(f"Fold: {fold + 1}")
            print(f"\ttrain_set len: {len(train_set)}")
            print(f"\tval_set len: {len(val_set)}")

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            # Define the optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=18, gamma=0.5)
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
                load_checkpoint=fold > 0,
            )

    validation_data = np.load(f"data/preprocessed_validation.npy")
    validation_data = torch.from_numpy(validation_data).float()

    validation_dataset = JigsawValidationDataset(validation_data)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    # Evaluate the model and save the results to a text file
    evaluate_model(model, validation_loader)
