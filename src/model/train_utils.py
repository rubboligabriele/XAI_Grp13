import os
import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def train(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits
        loss = criterion(logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Step the scheduler (if provided)
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(loss=loss.item(), acc=f"{100.0 * correct / total:.2f}%")

    loss_avg = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return loss_avg, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits
            loss = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss_avg = running_loss / len(loader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100.0

    # Compute confusion matrix (labels: 1 = melanoma, 0 = naevus)
    cm = confusion_matrix(all_labels, all_preds, labels=[1, 0])  # 1 = melanoma, 0 = naevus
    TP, FN, FP, TN = cm.ravel()

    # Compute precision, recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return loss_avg, balanced_acc, precision, recall, f1, cm, all_preds, all_labels


def train_loop(model, train_loader, optimizer, criterion, device,
               num_epochs, model_path, model_filename, scheduler=None):
    train_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        # Train for one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scheduler=scheduler)       
        train_losses.append(train_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    # Save the model after training
    os.makedirs(model_path, exist_ok=True)
    save_path = os.path.join(model_path, model_filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, train_losses