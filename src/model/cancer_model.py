import torch
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import AutoModelForImageClassification

import torch
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import AutoModelForImageClassification

def create_skin_cancer_model(dataset, learning_rate, freeze_backbone=False):
    model_name = "jhoppanne/SkinCancerClassifier_smote-V0"
    # Load the model from Hugging Face, and fine-tune it for binary classification
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # Freezing all layers except classifier if required
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")

    # Compute class weights to handle class imbalance in the dataset
    # 'balanced' computes weights inversely proportional to class frequencies
    targets = [label for _, label in dataset]
    class_weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Define loss and optimizer (only updating parameters with requires_grad=True)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    return model.to("cuda" if torch.cuda.is_available() else "cpu"), criterion, optimizer