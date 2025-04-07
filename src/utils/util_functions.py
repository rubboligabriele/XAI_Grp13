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
    parser.add_argument('--model_filename', type=str, default="Vitrans_melanoma_GOOD.pth", help="Name of the model file to load/save")
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for data loaders")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help="Directory where model weights are saved or loaded from")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help="Learning rate for the optimizer")
    parser.add_argument('--use_scheduler', action='store_true', help="Enable learning rate scheduler (linear)")
    return parser