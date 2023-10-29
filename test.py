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
def checkpoint_load(model, save_path, epoch, n_classes=0, model_ver=1):
    filename = "checkpoint-{:03d}.pth".format(epoch)
    f = os.path.join(save_path, filename)
    model.load_state_dict(torch.load(f))
    print("loaded checkpoint:", f, flush=True)


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
    filename = "data/test.txt"
    np.savetxt(filename, all_predictions, fmt="%d")

    # compress the results folder
    zip_filename = "data/test-result.zip"
    path = Path(zip_filename)
    if path.is_file():
        os.remove(zip_filename)
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(filename, arcname="test.txt")

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

# Parse the arguments
args = parser.parse_args()

num_classes = 50
batch_size = args.batch
checkpoint = args.checkpoint

print("checkpoint: ", checkpoint, "batch_size", batch_size)

if __name__ == "__main__":
    # Create the model
    # model = JigsawModel(n_classes=num_classes).to(device)
    model = JigsawNet(n_classes=num_classes).to(device)

    test_data = np.load(f"data/preprocessed_test.npy")
    test_data = torch.from_numpy(test_data).float()

    test_dataset = JigsawValidationDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    reset_random_generators()
    # Evaluate the model and save the results to a text file
    evaluate_model(model, test_loader, checkpoint=checkpoint)

# v4 submissions (20x data augmentation)
# checkpoint-002    32	0.422589
# checkpoint-003    29	0.427351
# checkpoint-004    33	0.431287
# checkpoint-005    34	0.428881
# checkpoint-007    31	0.416034
# checkpoint-008    30	0.414468
