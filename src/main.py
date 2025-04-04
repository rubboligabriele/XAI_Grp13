from data.preprocessing import get_dataloaders_skin
from model.cancer_model import create_skin_cancer_model
from model.train_utils import train, evaluate, EarlyStopping
from utils.plots import plot_loss, plot_confusion_matrix
from config import *
import os
import torch
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Train or evaluate skin cancer classification model")
parser.add_argument('--load_model', action='store_true', help="Load a pretrained model instead of training")
parser.add_argument('--model_filename', type=str, default="Vitrans_melanoma_GOOD.pth", help="Name of the model file to load/save")
parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help="Number of training epochs")
parser.add_argument('--patience', type=int, default=PATIENCE, help="Early stopping patience")
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for data loaders")
parser.add_argument('--model_path', type=str, default=MODEL_PATH, help="Directory where model weights are saved or loaded from")
parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help="Learning rate for the optimizer")
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, full_dataset_raw = get_dataloaders_skin(DATASET_PATH, args.batch_size)

model, criterion, optimizer = create_skin_cancer_model(full_dataset_raw, args.learning_rate)

#model_filename = "Vitrans_melanoma_GOOD.pth"
model_filename = args.model_filename

# Ask user whether to load pretrained model or train anew
#choice = input("Do you want to load a previously trained model? (y/n): ").strip().lower()

#if choice == 'y':
if args.load_model:
    model.load_state_dict(torch.load(os.path.join(args.model_path, model_filename), weights_only=True))
    print("Loaded pretrained model from saved_models.")
else:
    print("🛠️ Training new model...")
    # Early stopping setup
    #early_stopper = EarlyStopping(patience=PATIENCE)
    early_stopper = EarlyStopping(patience=args.patience)
    train_losses = []
    val_losses = []

    # Training loop
    #for epoch in range(NUM_EPOCHS):
        #print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{args.num_epochs}]")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_bal_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val Balanced Accuracy: {val_bal_acc:.2f}%")

        if early_stopper.step(val_bal_acc, model):
            print(f"\nEarly stopping triggered. Best Val Balanced Accuracy: {early_stopper.best_score:.2f}%")
            break

    model.load_state_dict(early_stopper.best_model_state)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_path, f"model_melanoma_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"))
    print("Best model saved to saved_models.")

    # Plot training curves
    plot_loss(train_losses, val_losses)

# Final evaluation
test_loss, test_bal_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}, Test Balanced Accuracy: {test_bal_acc:.2f}%")

# Plot results
plot_confusion_matrix(y_true, y_pred, full_dataset_raw.classes)
