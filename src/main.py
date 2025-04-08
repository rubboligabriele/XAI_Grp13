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
from transformers import get_scheduler

parser = get_parser()
args = parser.parse_args()
if args.compare_models and not args.second_model_filename:
    parser.error("--second_model_filename is required when --compare_models is set")

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

    if args.use_scheduler:
        total_training_steps = len(train_loader) * args.num_epochs
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_training_steps,
        )
    else:
        scheduler = None

    model, train_losses, val_losses = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion,
        device, args.num_epochs, args.patience,
        args.model_path, args.model_filename,
        scheduler=scheduler
    )
    plot_loss(train_losses, val_losses)

# Evaluation on test set
test_loss, test_bal_acc, precision, recall, f1, cm, y_pred, y_true = evaluate(model, test_loader, criterion, device)

print(f"\nTest Balanced Accuracy: {test_bal_acc:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

plot_confusion_matrix(cm, full_dataset_raw.classes)

# Patch for GradCAM
def forward_patch(x, **kwargs):
    return model.__class__.forward(model, pixel_values=x, **kwargs).logits

model.forward = forward_patch

deeplift = DeepLift(model)
ig = IntegratedGradients(model)

target_layer = model.efficientnet.encoder.top_conv
cam = GradCAM(model=model, target_layers=[target_layer])

original_images, overlays, deeplift_overlays, ig_overlays, titles = [], [], [], [], []
grayscale_cams, deeplift_attrs, ig_attrs = [], [], []

image_tensor = next(iter(test_loader))[0][:5]

for idx in range(5):
    input_image = image_tensor[idx].unsqueeze(0).to(device)

    # Denormalizzazione
    img_denorm = image_tensor[idx].clone()
    for t, m, s in zip(img_denorm, mean, std):
        t.mul_(s).add_(m)
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1).astype(np.float32)

    model.eval()
    with torch.no_grad():
        logits = model(input_image)
        pred_class_idx = logits.argmax(dim=1).item()
        pred_class_name = full_dataset_raw.classes[pred_class_idx]

    # GradCAM
    grayscale_cam = cam(input_tensor=input_image, targets=[ClassifierOutputTarget(pred_class_idx)])[0]
    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # DeepLIFT & IG
    input_image.requires_grad_()
    baseline = torch.zeros_like(input_image).to(device)
    dl_attr = deeplift.attribute(input_image, baselines=baseline, target=pred_class_idx)
    ig_attr = ig.attribute(input_image, baselines=baseline, target=pred_class_idx, n_steps=200)

    original_images.append(img_np)
    overlays.append(overlay)
    deeplift_overlays.append(attribution_to_grayscale(dl_attr))
    ig_overlays.append(attribution_to_grayscale(ig_attr))
    titles.append(pred_class_name)

    grayscale_cams.append(grayscale_cam)
    deeplift_attrs.append(dl_attr.squeeze().detach().cpu().numpy())
    ig_attrs.append(ig_attr.squeeze().detach().cpu().numpy())

plot_explainability_comparison(original_images, overlays, deeplift_overlays, ig_overlays, titles)

ssim_rows = []
pearson_rows = []
cosine_rows = []
image_labels = []

for i in range(5):
    cam_gray = grayscale_cams[i]
    dl = deeplift_overlays[i]
    ig = ig_overlays[i]

    cam_flat = grayscale_cams[i].flatten()
    dl_flat = deeplift_overlays[i].flatten()
    ig_flat = ig_overlays[i].flatten()

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

if args.compare_models:
    model2, _, _ = create_skin_cancer_model(full_dataset_raw, args.learning_rate)
    model2.load_state_dict(torch.load(os.path.join(args.model_path, args.second_model_filename), weights_only=True))

    def forward_patch_2(x, **kwargs):
        return model2.__class__.forward(model2, pixel_values=x, **kwargs).logits
    model2.forward = forward_patch_2

    deeplift2 = DeepLift(model2)
    ig2 = IntegratedGradients(model2)
    target_layer2 = model2.efficientnet.encoder.top_conv
    cam2 = GradCAM(model=model2, target_layers=[target_layer2])

    grayscale_cams_2, deeplift_attrs_2, ig_attrs_2 = [], [], []

    for idx in range(5):
        input_image = image_tensor[idx].unsqueeze(0).to(device)
        logits2 = model2(input_image)
        pred_class_idx2 = logits2.argmax(dim=1).item()

        model2.eval()
        logits2 = model2(input_image)
        pred_class_idx2 = logits2.argmax(dim=1).item()

        grayscale_cam2 = cam2(input_tensor=input_image, targets=[ClassifierOutputTarget(pred_class_idx2)])[0]
        dl_attr2 = deeplift2.attribute(input_image, baselines=baseline, target=pred_class_idx2)
        ig_attr2 = ig2.attribute(input_image, baselines=baseline, target=pred_class_idx2, n_steps=200)

        grayscale_cams_2.append(grayscale_cam2)
        deeplift_attrs_2.append(dl_attr2.squeeze().detach().cpu().numpy())
        ig_attrs_2.append(ig_attr2.squeeze().detach().cpu().numpy())

    jaccard_rows = []
    for i in range(5):
        jaccard_cam = compute_jaccard_similarity(grayscale_cams[i], grayscale_cams_2[i])
        jaccard_dl = compute_jaccard_similarity(deeplift_attrs[i], deeplift_attrs_2[i])
        jaccard_ig = compute_jaccard_similarity(ig_attrs[i], ig_attrs_2[i])
        jaccard_rows.append([jaccard_cam, jaccard_dl, jaccard_ig])

    jaccard_df = pd.DataFrame(jaccard_rows, columns=["GradCAM", "DeepLIFT", "IG"], index=[f"Img {i+1}" for i in range(5)])
    plot_jaccard_heatmap(jaccard_df)