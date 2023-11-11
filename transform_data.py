from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


def calculate_distance(key_p, p):
    similarity = cosine_similarity([key_p], [p])
    distance = 1 - similarity
    distance = distance[0][0]
    return distance


def distance_transform(data):
    distance_data = []
    for puzzle in tqdm(data):
        puzzle_data = []
        for image in puzzle:
            image_vector = [calculate_distance(image, p) for p in puzzle]
            image_vector = image_vector / sum(image_vector)  # normalize each row
            puzzle_data.append(image_vector)
        distance_data.append(puzzle_data)

    return distance_data


class JigsawModel(nn.Module):
    def __init__(
        self,
        num_classes,
        include_softmax=False,
    ):
        super(JigsawModel, self).__init__()
        self.include_softmax = include_softmax

        self.fc1 = nn.Linear(36 * 36, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.bn4 = nn.BatchNorm1d(num_classes)  # Batch normalization after fc4

    def forward(self, x):
        x = x.view(-1, 36 * 36)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(
            self.bn4(self.fc4(x))
        )  # Apply batch normalization after fc4 and before activation

        if self.include_softmax:
            x = F.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    folders = ["train", "validation", "test"]

    for folder in folders:
        # Define the directory where your images are located
        data_dir = f"data/{folder}"
        print("data_dir: ", data_dir)

        fold = 21 if "train" == folder else 1

        basename = f"timm_preprocessed_{folder}"
        for i in range(fold):
            filename = (
                f"data/distance_{basename}_{i - 1}.npy"
                if i > 0
                else f"data/distance_{basename}.npy"
            )
            path = Path(filename)
            if path.is_file():
                print(f"File {filename} exists - skipping ...", flush=True)
                continue

            preprocessed_filename = (
                f"data/{basename}_{i - 1}.npy" if i > 0 else f"data/{basename}.npy"
            )

            preprocessed_data = np.load(preprocessed_filename)

            data = distance_transform(preprocessed_data)

            np.save(filename, data)
            print(f"saved file: {filename}", flush=True)
