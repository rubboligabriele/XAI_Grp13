from data.preprocessing import get_dataloaders_skin
from model.cancer_model import create_skin_cancer_model
from model.train_utils import train, evaluate, EarlyStopping
from utils.plots import plot_loss, plot_confusion_matrix, plot_explainability_comparison
from config import *
import os
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from captum.attr import DeepLift, IntegratedGradients
from utils.util_functions import *
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from skimage.color import rgb2gray
import pandas as pd

args = get_parser().parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, full_dataset_raw, mean, std = get_dataloaders_skin(DATASET_PATH, args.batch_size)

model, criterion, optimizer = create_skin_cancer_model(full_dataset_raw, args.learning_rate)

model_filename = args.model_filename

if args.load_model:
    model.load_state_dict(torch.load(os.path.join(args.model_path, model_filename), weights_only=True))
    print("Loaded pretrained model.")
else:
    print("Training new model...")
    # Early stopping setup
    early_stopper = EarlyStopping(patience=args.patience)
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{args.num_epochs}]")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_bal_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val Balanced Accuracy: {val_bal_acc:.2f}%")

        if early_stopper.step(-val_loss, model):
            print(f"\nEarly stopping triggered. Best Val Loss: {-early_stopper.best_score:.4f}")
            break

    model.load_state_dict(early_stopper.best_model_state)

    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_path, args.model_filename))
    print(f"Best model saved to {os.path.join(args.model_path, args.model_filename)}")

    # Plot training curves
    plot_loss(train_losses, val_losses)

# Final evaluation
test_loss, test_bal_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}, Test Balanced Accuracy: {test_bal_acc:.2f}%")

# Plot results
plot_confusion_matrix(y_true, y_pred, full_dataset_raw.classes)

# Patch for GradCAM
def forward_patch(x, **kwargs):
    return model.__class__.forward(model, pixel_values=x, **kwargs).logits

model.forward = forward_patch

deeplift = DeepLift(model)
ig = IntegratedGradients(model)

target_layer = model.efficientnet.encoder.top_conv
cam = GradCAM(model=model, target_layers=[target_layer])

original_images = []
overlays = []
deeplift_overlays = []
ig_overlays = []
titles = []

image_tensor = next(iter(test_loader))[0][:5]

for idx in range(5):
    input_image = image_tensor[idx].unsqueeze(0).to(device)

    # Denormalizzazione
    img_denorm = image_tensor[idx].clone()
    for t, m, s in zip(img_denorm, mean, std):
        t.mul_(s).add_(m)
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1).astype(np.float32)

    # Prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_image)
        pred_class_idx = logits.argmax(dim=1).item()
        pred_class_name = full_dataset_raw.classes[pred_class_idx]

    # GradCAM
    grayscale_cam = cam(input_tensor=input_image, targets=[ClassifierOutputTarget(pred_class_idx)])[0]
    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # DeepLIFT
    input_image.requires_grad_()
    baseline = torch.zeros_like(input_image).to(device)
    dl_attr = deeplift.attribute(input_image, baselines=baseline, target=pred_class_idx)
    dl_overlay = attribution_to_grayscale(dl_attr)

    #Integrated Gradients
    ig_attr = ig.attribute(input_image, baselines=baseline, target=pred_class_idx, n_steps=200)
    ig_overlay = attribution_to_grayscale(ig_attr)

    original_images.append(img_np)
    overlays.append(overlay)
    deeplift_overlays.append(dl_overlay)
    ig_overlays.append(ig_overlay)
    titles.append(pred_class_name)

plot_explainability_comparison(original_images, overlays, deeplift_overlays, ig_overlays, titles)

ssim_rows = []
pearson_rows = []
cosine_rows = []
image_labels = []

for i in range(5):
    dl = deeplift_overlays[i]
    ig = ig_overlays[i]
    cam_rgb = overlays[i]
    cam_gray = rgb2gray(cam_rgb)

    # Flatten
    dl_flat = dl.flatten()
    ig_flat = ig.flatten()
    cam_flat = cam_gray.flatten()

    image_label = f"Img {i+1}"
    image_labels.append(image_label)

    ssim_rows.append([
        ssim(cam_gray, dl, data_range=1.0),
        ssim(cam_gray, ig, data_range=1.0),
        ssim(dl, ig, data_range=1.0)
    ])

    pearson_rows.append([
        pearsonr(cam_flat, dl_flat)[0],
        pearsonr(cam_flat, ig_flat)[0],
        pearsonr(dl_flat, ig_flat)[0]
    ])

    cosine_rows.append([
        cosine_similarity([cam_flat], [dl_flat])[0][0],
        cosine_similarity([cam_flat], [ig_flat])[0][0],
        cosine_similarity([dl_flat], [ig_flat])[0][0]
    ])
columns = ["CAM-DL", "CAM-IG", "DL-IG"]

ssim_df = pd.DataFrame(ssim_rows, columns=columns, index=image_labels)
pearson_df = pd.DataFrame(pearson_rows, columns=columns, index=image_labels)
cosine_df = pd.DataFrame(cosine_rows, columns=columns, index=image_labels)