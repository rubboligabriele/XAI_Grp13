import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from config import LEARNING_RATE
from transformers import AutoModelForImageClassification

def create_model(dataset):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    for name, param in model.named_parameters():
        param.requires_grad = any(x in name for x in ["denseblock4", "norm5", "classifier"])
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    # Class Reweighing
    targets = [label for _, label in dataset]
    weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
    weights = torch.tensor(weights, dtype=torch.float)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    return model.to("cuda" if torch.cuda.is_available() else "cpu"), criterion, optimizer

def create_skin_cancer_model(dataset):
    model_name = "Anwarkh1/Skin_Cancer-Image_Classification"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    num_classes = 2
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    targets = [label for _, label in dataset]
    class_weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    return model.to("cuda" if torch.cuda.is_available() else "cpu"), criterion, optimizer