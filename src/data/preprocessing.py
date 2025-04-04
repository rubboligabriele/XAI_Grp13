from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import numpy as np
from transformers import AutoImageProcessor
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.data[idx]
        # Pass the image directly to the transform function without using keyword 'images='
        encoding = self.transform(img)  # Corrected here: just pass img
        return encoding["pixel_values"].squeeze(0), label  # Remove batch dimension

    def __len__(self):
        return len(self.data)

def get_dataloaders_skin(dataset_path, batch_size):
    dataset = datasets.ImageFolder(dataset_path)
    feature_extractor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification", use_fast=True)

    def transform_image(image):
        # Use feature extractor to transform the image, ensuring it's in the correct format
        return feature_extractor(image, return_tensors="pt")

    targets = np.array([label for _, label in dataset])

    # Stratified split
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(split_1.split(np.zeros(len(targets)), targets))
    train_val_targets = targets[train_val_idx]

    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    train_idx, val_idx = next(split_2.split(np.zeros(len(train_val_targets)), train_val_targets))

    train_data = [dataset[i] for i in train_val_idx]
    val_data = [train_data[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    train_ds = CustomDataset(train_data, transform=transform_image)
    val_ds = CustomDataset(val_data, transform=transform_image)
    test_ds = CustomDataset(test_data, transform=transform_image)

    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(val_ds, batch_size, shuffle=False),
        DataLoader(test_ds, batch_size, shuffle=False),
        dataset
    )