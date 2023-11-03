import os
import random
import timm
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

print("device: ", device)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

folders = ["train", "validation", "test"]

for folder in folders:
    # Define the directory where your images are located
    data_dir = f"data/{folder}"
    print("data_dir: ", data_dir)

    is_training = "train" == folder

    # Load a pretrained ResNet-50 model
    pretrained_model = timm.create_model(
        "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    data_config = timm.data.resolve_model_data_config(pretrained_model)
    transforms = timm.data.create_transform(**data_config, is_training=is_training)

    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    # Preprocess the images and extract image vectors
    def preprocess_and_extract_vectors(image_path):
        image = Image.open(image_path).convert("RGB")
        # get model specific transforms (normalization, resize)
        image = transforms(image)

        # Extract image vector using the pretrained model
        image_vector = extract_image_vector(image)
        return image_vector

    def extract_image_vector(image):
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            image_vector = pretrained_model(image)
        image_vector = image_vector.view(-1)
        return image_vector.cpu().numpy()

    # Define the dimensions of your images
    image_width, image_height = 56, 56

    # Define the total number of data points and images per data point
    images_per_data_point = 36
    total_data_points = int(len(os.listdir(data_dir)) / images_per_data_point)
    print("total_data_points: ", total_data_points)

    fold = 20 if is_training else 1

    for i in range(fold):
        filename = (
            f"data/timm_preprocessed_{folder}_{i}.npy"
            if is_training
            else f"data/timm_preprocessed_{folder}.npy"
        )

        path = Path(filename)
        if path.is_file():
            print(f"File {filename} exists - skipping ...", flush=True)
            continue

        print(f"File {filename} does not exist - creating a new one ...", flush=True)

        # # Initialize an empty NumPy array to store the data
        data = np.empty(
            (total_data_points, images_per_data_point, 3072), dtype=np.float32
        )  # Assuming ResNet-50 outputs 2048-dimensional vectors

        # Loop through each data point
        for data_point_index in range(total_data_points):
            # Loop through each image within a data point
            for image_index in range(images_per_data_point):
                # Construct the image file path
                image_filename = f"{data_point_index}_{image_index}.jpg"
                image_path = os.path.join(data_dir, image_filename)

                # Check if the image file exists
                if os.path.exists(image_path):
                    # Preprocess the image and extract image vector
                    image_vector = preprocess_and_extract_vectors(image_path)
                    # Store the image vector in the data array
                    data[data_point_index, image_index] = image_vector

        np.save(filename, data)
        print(f"saved file: {filename}", flush=True)
