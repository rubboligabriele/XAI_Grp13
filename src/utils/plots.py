import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def plot_loss(train, val):
    plt.figure(figsize=(8, 5))
    plt.plot(train, label="Train Loss", marker="o")
    plt.plot(val, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, vmin=0, vmax=15):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    disp.im_.set_clim(vmin, vmax)
    plt.title("Confusion Matrix (Test Set)")
    plt.show()

def plot_explainability_comparison(originals, cam_overlays, dl_overlays, ig_overlays, pred_classes):
    fig, axes = plt.subplots(len(originals), 4, figsize=(16, 4 * len(originals)))

    for i in range(len(originals)):
        axes[i, 0].imshow(originals[i])
        axes[i, 0].set_title(f"Pred: {pred_classes[i]}")
        axes[i, 1].imshow(cam_overlays[i])
        axes[i, 1].set_title("GradCAM")
        axes[i, 2].imshow(dl_overlays[i], cmap='gray')
        axes[i, 2].set_title("DeepLIFT")
        axes[i, 3].imshow(ig_overlays[i], cmap='gray')
        axes[i, 3].set_title("Integrated Gradients")

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_similarity_heatmaps(ssim_df, pearson_df, cosine_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ssim_vmin, ssim_vmax = -0.1, 0.4
    pearson_vmin, pearson_vmax = -0.7, 0.6
    cosine_vmin, cosine_vmax = 0.4, 0.9

    sns.heatmap(ssim_df, annot=True, cmap="Blues", fmt=".2f", ax=axes[0],
            vmin=ssim_vmin, vmax=ssim_vmax)
    axes[0].set_title("SSIM Similarity")
    axes[0].set_xlabel("Pair")
    axes[0].set_ylabel("Image")
    axes[0].tick_params(axis='y', labelrotation=0)

    sns.heatmap(pearson_df, annot=True, cmap="Reds", fmt=".2f", ax=axes[1],
            vmin=pearson_vmin, vmax=pearson_vmax)
    axes[1].set_title("Pearson Correlation")
    axes[1].set_xlabel("Pair")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis='y', labelrotation=0)

    sns.heatmap(cosine_df, annot=True, cmap="Greens", fmt=".2f", ax=axes[2],
            vmin=cosine_vmin, vmax=cosine_vmax)
    axes[2].set_title("Cosine Similarity")
    axes[2].set_xlabel("Pair")
    axes[2].set_ylabel("")
    axes[2].tick_params(axis='y', labelrotation=0)

    plt.tight_layout()
    plt.show()