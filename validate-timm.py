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
import timm
from transform_data import JigsawModel
from train_v9 import *


class JigsawValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        return puzzle


def evaluate_model(model, data_loader, checkpoint=-1):
    save_path = os.path.join(os.getcwd(), "data", "checkpoints/")

    if checkpoint < 0:
        for _, _, files in os.walk(save_path):
            for filename in files:
                cp = int(re.split("[-.]", filename)[-2])
                if cp > checkpoint:
                    checkpoint = cp

    checkpoint_load(model, save_path, checkpoint)

    model.eval()
    data, labels = load_training_data(0, max_fold=0)
    val_set = JigsawDataset(data, labels)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    all_predictions = []  # To store translated predictions

    with torch.no_grad():
        model_result = []
        total_targets = []

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            model_batch_result = model(inputs)

            model_result.extend(model_batch_result.cpu().numpy())
            total_targets.extend(targets.cpu().numpy())

        accuracy = timm.utils.accuracy(
            torch.from_numpy(np.array(model_result)),
            torch.from_numpy(np.array(total_targets)),
            topk=(1,),
        )[0]
        print(
            "checkpoint:{:3d} - final accuracy:{:.3f}%".format(
                checkpoint,
                accuracy,
            ),
            flush=True,
        )

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
    filename = f"data/{name}.txt"
    np.savetxt(filename, all_predictions, fmt="%d")

    # compress the results folder
    zip_filename = f"data/{name}-result.zip"
    path = Path(zip_filename)
    if path.is_file():
        os.remove(zip_filename)
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(filename, arcname=f"{name}.txt")

    print(f"results saved to: {zip_filename}")


def reset_random_generators():
    RANDOM_SEED = 193

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

print(f"device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=64)
parser.add_argument(
    "-c",
    "--checkpoint",
    type=int,
    help="checkpoint to evaluate",
    default=-1,
)

parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="name for input/output",
    default="validation",
)

# Parse the arguments
args = parser.parse_args()

num_classes = 50
batch_size = args.batch
checkpoint = args.checkpoint
name = args.name

print("checkpoint: ", checkpoint, "batch_size: ", batch_size, "name: ", name)

if __name__ == "__main__":
    # Create the model
    model = JigsawModel(num_classes=num_classes).to(device)
    # model = JigsawNet(
    # n_classes=num_classes, num_features=3072, relu_in_last_fc=True
    # ).to(device)

    filename = f"data/distance_timm_preprocessed_{name}.npy"
    validation_data = np.load(filename)
    validation_data = torch.from_numpy(validation_data).float()

    validation_dataset = JigsawValidationDataset(validation_data)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"loaded validation data from: {filename}")

    reset_random_generators()
    # Evaluate the model and save the results to a text file
    evaluate_model(model, validation_loader, checkpoint=checkpoint)
