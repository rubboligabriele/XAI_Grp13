from data.preprocessing import get_dataloaders_skin
from model.cancer_model import create_skin_cancer_model
from model.train_utils import *
from utils.plots import *
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
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_filename), weights_only=True))
    print("Loaded pretrained model.")
else:
    print("Training new model...")
    model, train_losses, val_losses = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion,
        device, args.num_epochs, args.patience,
        args.model_path, args.model_filename
    )
    plot_loss(train_losses, val_losses)

# Evaluation on test set
test_loss, test_bal_acc, detection_rate, neg_precision, cm, y_pred, y_true = evaluate(model, test_loader, criterion, device)

print(f"\nTest Balanced Accuracy: {test_bal_acc:.2f}%")
print(f"Detection Rate (TN/N): {detection_rate:.2f}%")
print(f"Negative Precision (TN/PN): {neg_precision:.2f}%")

plot_confusion_matrix(cm, full_dataset_raw.classes)

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

plot_similarity_heatmaps(ssim_df, pearson_df, cosine_df)