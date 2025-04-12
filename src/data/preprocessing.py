from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import numpy as np
from transformers import AutoImageProcessor

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        """
        Initialize the CustomDataset.

        Parameters:
        - data: List of tuples containing (image, label)
        - transform: Transformation function to apply to the images
        """
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieve an image-label pair from the dataset and apply transformation.

        Parameters:
        - idx: Index of the data point

        Returns:
        - Transformed image tensor and corresponding label
        """
        img, label = self.data[idx]
        encoding = self.transform(img)
        return encoding["pixel_values"].squeeze(0), label

    def __len__(self):
        return len(self.data)

def get_dataloaders_skin(dataset_path, batch_size):
    """
    Retrieve the train and test datasets and corresponding DataLoaders, 
    with no validation set.

    Parameters:
    - dataset_path: Path to the dataset directory
    - batch_size: Batch size for the DataLoader

    Returns:
    - train_loader: DataLoader for the training set
    - test_loader: DataLoader for the test set
    - dataset: The full dataset
    - mean: The mean of the dataset for normalization
    - std: The standard deviation of the dataset for normalization
    """
    dataset = datasets.ImageFolder(dataset_path)
    feature_extractor = AutoImageProcessor.from_pretrained("jhoppanne/SkinCancerClassifier_smote-V0")
    mean = feature_extractor.image_mean
    std = feature_extractor.image_std

    def transform_image(image):
        # Use feature extractor to transform the image, ensuring it's in the correct format
        return feature_extractor(image, return_tensors="pt")

    targets = np.array([label for _, label in dataset])

    # Stratified split to ensure balanced class distribution in train and test sets
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(split.split(np.zeros(len(targets)), targets))

    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]

    train_ds = CustomDataset(train_data, transform=transform_image)
    test_ds = CustomDataset(test_data, transform=transform_image)

    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(test_ds, batch_size, shuffle=False),
        dataset,
        mean,
        std
    )