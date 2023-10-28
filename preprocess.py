import os
import cv2
import random
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

input_size = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess_image = A.Compose(
    [
        A.SmallestMaxSize(max_size=input_size + 48),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=input_size, width=input_size),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ]
)

# folders = ["train", "validation"]
folders = ["train"]

for folder in folders:
    # Define the directory where your images are located
    data_dir = f"data/{folder}"
    print("data_dir: ", data_dir)

    # Load a pretrained ResNet-50 model
    pretrained_model = models.resnet50(
        weights="ResNet50_Weights.DEFAULT"
    )  # Use updated weights argument
    pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    # Preprocess the images and extract image vectors
    def preprocess_and_extract_vectors(image_path, preprocess=None):
        # Preprocessing transforms
        if preprocess is None:
            image = Image.open(image_path).convert("RGB")
            preprocess = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224)
                    ),  # Resize to match the pretrained model's input size
                    transforms.ToTensor(),  # Convert image to tensor
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # Normalize
                ]
            )
            image = preprocess(image)
        else:
            # use cv2 for albumentation
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess(image=image)["image"]

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

    fold = 1
    preporcess = None
    if folder == "train":
        fold = 10
        preporcess = preprocess_image

    for i in range(fold):
        # # Initialize an empty NumPy array to store the data
        data = np.empty(
            (total_data_points, images_per_data_point, 2048), dtype=np.float32
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
                    image_vector = preprocess_and_extract_vectors(
                        image_path, preporcess
                    )
                    # Store the image vector in the data array
                    data[data_point_index, image_index] = image_vector

        filename = f"data/preprocessed_{folder}.npy"
        if fold > 0:
            filename = f"data/preprocessed_{folder}_{i}.npy"
        np.save(filename, data)
        print(f"saved file: {filename}", flush=True)
