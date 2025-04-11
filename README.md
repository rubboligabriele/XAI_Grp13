# XAI_Grp13

## Motivation and Description

This project explores explainability in deep learning for melanoma classification. We compare different XAI methods (GradCAM, DeepLIFT, Integrated Gradients) applied to an EfficientNet-based model, pre-trained on dermoscopic images.
We fine-tune this model on a small dataset (170 samples) of dermoscopic images collected at UMCG, focusing on binary classification (melanoma vs. benign).

The aim is to obtain a reliable and calibrated classifier and evaluate how stable and trustworthy the explanations are, both within the same model and across different models. We analyze the alignment between methods by comparing their explanation heatmaps using multiple similarity metrics (SSIM, Pearson correlation, Cosine similarity, and Jaccard index). This allows us to assess whether different XAI techniques agree on which regions of the image are most relevant for the model’s prediction, and whether such explanations remain consistent across models with similar performance but different generalization behaviors.

Through both qualitative visualizations and quantitative metrics, we aim to provide insights into the robustness and interpretability of deep learning models in the context of medical diagnosis.

---

## Main Functionalities

- Binary classification of skin lesions (melanoma vs. naevus)
- Training/fine-tuning and evaluation of a EfficientNet model
- Explanation of predictions via GradCAM, DeepLIFT, and Integrated Gradients
- Visual and quantitative comparison of the explanation heatmaps
- Similarity metrics: SSIM, Pearson Correlation, Cosine Similarity, Jaccard Index
- Model comparison mode to assess robustness and comparability of explanations across different models

---

## Folder Structure

```bash
.
├── README.md
├── complete_mednode_dataset
│   ├── melanoma
│   └── naevus
├── requirements.txt
└── src
    ├── config.py
    ├── data
    │   └── preprocessing.py
    ├── main.py
    ├── model
    │   ├── cancer_model.py
    │   └── train_utils.py
    └── utils
        ├── plots.py
        └── util_functions.py
```


## Setup Environment

You can set up the project environment using **pip**.

1. Create and activate a virtual environment:
   ```bash
   python -m venv xai_env
   source xai_env/bin/activate       # On Windows: xai_env\Scripts\activate
   ```
2. Install all dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu       # Torch for CPU
   pip install -r requirements.txt
   ```
