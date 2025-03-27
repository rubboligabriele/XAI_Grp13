import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score


def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits  # Estrai i logits dal risultato del modello
        loss = criterion(logits, labels)  # Usa i logits per il calcolo della loss
        loss.backward()
        optimizer.step()

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
            logits = outputs.logits  # Estrai i logits dal risultato del modello
            loss = criterion(logits, labels)  # Usa i logits per il calcolo della loss

            running_loss += loss.item()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss_avg = running_loss / len(loader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100.0
    return loss_avg, balanced_acc, all_preds, all_labels

class EarlyStopping:
    def __init__(self, patience=3):
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