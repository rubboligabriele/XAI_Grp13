import torch
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import AutoModelForImageClassification

def create_skin_cancer_model(dataset, learning_rate):
    model_name = "Anwarkh1/Skin_Cancer-Image_Classification"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    num_classes = 2
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    targets = [label for _, label in dataset]
    class_weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)

    return model.to("cuda" if torch.cuda.is_available() else "cpu"), criterion, optimizer