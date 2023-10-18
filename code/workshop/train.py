import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

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

# Ensure shuffle = False when evaluating on validation and test
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


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

    def forward(self, x):
        x = x.view(-1, 36 * 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(
            self.bn4(self.fc4(x))
        )  # Apply batch normalization after fc4 and before activation
        x = F.softmax(x, dim=1)
        return x


# Create the model
model = JigsawModel(num_positions=num_classes).to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train_model(model, train_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for puzzle, label in train_loader:
            puzzle, label = puzzle.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(puzzle)
            loss = criterion(outputs.float(), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted == label).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%"
        )


# Train the model
train_model(model, train_loader, optimizer, num_epochs)


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
            output = model(puzzle)
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
