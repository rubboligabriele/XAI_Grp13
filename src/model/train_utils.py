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

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

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

    cm = confusion_matrix(all_labels, all_preds, labels=[1, 0])  # 1 = melanoma (positivo), 0 = naevus (negativo)
    TP, FN, FP, TN = cm.ravel()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return loss_avg, balanced_acc, precision, recall, f1, cm, all_preds, all_labels

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.best_model_state = None

    def step(self, val_score, model):
        if self.best_score is None or val_score > self.best_score:
            self.best_score = val_score
            self.best_model_state = model.state_dict()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        
def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, device,
                              num_epochs, patience, model_path, model_filename, scheduler=None):
    early_stopper = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scheduler=scheduler)
        val_loss, val_bal_acc, *_ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val Balanced Accuracy: {val_bal_acc:.2f}%")

        if early_stopper.step(-val_loss, model):
            print(f"\nEarly stopping triggered. Best Val Loss: {-early_stopper.best_score:.4f}")
            break

    # Load best weights and save
    model.load_state_dict(early_stopper.best_model_state)
    os.makedirs(model_path, exist_ok=True)
    save_path = os.path.join(model_path, model_filename)
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")

    return model, train_losses, val_losses