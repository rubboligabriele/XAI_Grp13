import numpy as np
import argparse
from config import *

def attribution_to_grayscale(attr_tensor, gamma=0.5):
    attr = attr_tensor.squeeze().detach().cpu().numpy()
    if attr.ndim == 3:
        attr = np.abs(attr).mean(axis=0)
    else:
        attr = np.abs(attr)
    
    # Normalizing between 0 and 1
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    
    # Contrast augmentation
    attr = attr ** gamma

    return attr

def get_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate skin cancer classification model")
    parser.add_argument('--load_model', action='store_true', help="Load a pretrained model instead of training")
    parser.add_argument('--model_filename', type=str, required=True, help="Name of the model file to load/save")
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for data loaders")
    parser.add_argument('--model_path', type=str, required=True, help="Directory where model weights are saved or loaded from")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help="Learning rate for the optimizer")
    parser.add_argument('--use_scheduler', action='store_true', help="Enable learning rate scheduler (linear)")
    parser.add_argument('--compare_models', action='store_true', help="Compare two models using Jaccard similarity on explanations")
    parser.add_argument('--second_model_filename', type=str, help="Filename of the second model to compare")
    return parser

def compute_jaccard_similarity(map1, map2, percentile=90):
    def normalize(m):
        m = m - np.min(m)
        if np.max(m) != 0:
            m = m / np.max(m)
        return m

    def binarize(m, p):
        threshold = np.percentile(m, p)
        return (m >= threshold).astype(int)

    map1 = normalize(map1)
    map2 = normalize(map2)

    mask1 = binarize(map1, percentile)
    mask2 = binarize(map2, percentile)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0
    return intersection / union