import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=20)
parser.add_argument("-b", "--batch", type=int, help="Batch size", default=32)

# Parse the arguments
args = parser.parse_args()
print(device)

num_classes = 50
batch_size = args.batch
num_epochs = args.epochs

print("classes: ", num_classes, "batch", batch_size)

# Define the dimensions of your images
image_width, image_height = 56, 56

# Define the total number of data points and images per data point
total_data_points = 2944
images_per_data_point = 36

preprocessed_data = np.load(f"data/preprocessed_train.npy")
# Convert the NumPy array to PyTorch tensors
data = torch.from_numpy(preprocessed_data).float()


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


# Load the labels
labels = np.loadtxt(f"data/train/label_train.txt")
labels = torch.from_numpy(labels).long()

# Define the dataset and dataloader
dataset = JigsawDataset(data, labels)
print(f"dataset len: {len(dataset)}")

train_set, val_set = random_split(dataset, [0.8, 0.2])
print(f"train_set len: {len(train_set)}")
print(f"val_set len: {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# # Define the model architecture
class JigsawModel(nn.Module):
    def __init__(self, num_positions):
        super(JigsawModel, self).__init__()
        self.num_positions = num_positions
        self.fc1 = nn.Linear(36 * 2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 50)
        self.bn4 = nn.BatchNorm1d(50)  # Batch normalization after fc4

        self.fc5 = nn.Linear(128, 36)
        self.bn5 = nn.BatchNorm1d(36)  # Batch normalization after fc5

    def forward(self, x):
        x = x.view(-1, 36 * 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        c = F.relu(
            self.bn4(self.fc4(x))
        )  # Apply batch normalization after fc4 and before activation
        c = F.softmax(c, dim=1)

        p = None

        if self.training:
            p = F.relu(self.bn5(self.fc5(x)))
            p = torch.argsort(p)

        return c, p


class JigsawLoss(nn.Module):
    def __init__(self, device, alpha=0.5):
        super().__init__()
        pos_labels = [i for i in range(36)]
        self.pos_labels = torch.FloatTensor(pos_labels).to(device)
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        perm_pred, pos_pred = outputs

        loss = self.criterion(perm_pred.float(), labels)

        if pos_pred is not None:
            loss = (1 - self.alpha) * loss

            position_loss = 0

            for pos in pos_pred:
                position_loss += self.criterion(pos.float(), self.pos_labels)

            loss += self.alpha * position_loss

        return loss


# Create the model
model = JigsawModel(num_positions=num_classes).to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = JigsawLoss(device)


def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    highest_accuracy = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for puzzle, label in train_loader:
            puzzle, label = puzzle.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(puzzle)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs[0], 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted == label).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%"
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
                _, predicted = torch.max(outputs[0], 1)
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

    print(
        f"Best epoch {best_epoch + 1}, Highest Validation Accuracy: {highest_accuracy * 100:.2f}%",
        flush=True,
    )


# Train the model
train_model(model, train_loader, val_loader, optimizer, num_epochs)


class JigsawValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        return puzzle


def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []  # To store translated predictions

    with torch.no_grad():
        for puzzle in data_loader:
            puzzle = puzzle.to(device)
            output, _ = model(puzzle)
            _, predicted = torch.max(
                output, 1
            )  # Get the index of the max log-probability
            all_predictions.extend(predicted.cpu().detach().numpy())

    all_predictions = np.array(all_predictions)
    all_predictions = all_predictions.astype(int)

    # Save the predicted values to a text file
    np.savetxt("data/validation.txt", all_predictions, fmt="%d")


validation_data = np.load(f"data/preprocessed_validation.npy")
validation_data = torch.from_numpy(validation_data).float()

validation_dataset = JigsawValidationDataset(validation_data)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model and save the results to a text file
evaluate_model(model, validation_loader)
