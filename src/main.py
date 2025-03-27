from data.preprocessing import get_dataloaders_skin
from model.densenet import create_skin_cancer_model
from model.train_utils import train, evaluate, EarlyStopping
from utils.plots import plot_loss, plot_confusion_matrix
from config import *
import os
import torch
from datetime import datetime

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, full_dataset_raw = get_dataloaders_skin(DATASET_PATH)

model, criterion, optimizer = create_skin_cancer_model(full_dataset_raw)

model_filename = "Vitrans_melanoma_GOOD.pth"

# Ask user whether to load pretrained model or train anew
choice = input("Do you want to load a previously trained model? (y/n): ").strip().lower()

if choice == 'y':
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_filename)))
    print("Loaded pretrained model from saved_models.")
else:
    print("🛠️ Training new model...")
    # Early stopping setup
    early_stopper = EarlyStopping(patience=PATIENCE)
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_bal_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val Balanced Accuracy: {val_bal_acc:.2f}%")

        if early_stopper.step(val_bal_acc, model):
            print(f"\nEarly stopping triggered. Best Val Balanced Accuracy: {early_stopper.best_score:.2f}%")
            break

    # Load best model
    model.load_state_dict(early_stopper.best_model_state)

    # Save best model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"model_melanoma_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"))
    print("Best model saved to saved_models.")

    # Plot training curves
    plot_loss(train_losses, val_losses)

# Final evaluation
test_loss, test_bal_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}, Test Balanced Accuracy: {test_bal_acc:.2f}%")

# Plot results
plot_confusion_matrix(y_true, y_pred, full_dataset_raw.classes)
