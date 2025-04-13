# XAI_Grp13

## Motivation and Description

This project explores explainability in deep learning for melanoma classification. We compare different XAI methods (GradCAM, DeepLIFT, Integrated Gradients) applied to an EfficientNet-based model, pre-trained on dermoscopic images.
We fine-tune this model on a small dataset (170 samples) of non-dermoscopic images collected at UMCG, focusing on binary classification (melanoma vs. benign).

The aim is to obtain a reliable and calibrated classifier and evaluate how stable and trustworthy the explanations are, both within the same model and across different models. We analyze the alignment between methods by comparing their explanation heatmaps using multiple similarity metrics (SSIM, Pearson correlation, Cosine similarity, and Jaccard similarity). This allows us to assess whether different XAI techniques agree on which regions of the image are most relevant for the model’s prediction, and whether such explanations remain consistent across models with similar performance but different generalization behaviors.

Through both qualitative visualizations and quantitative metrics, we aim to provide insights into the robustness and interpretability of deep learning models in the context of medical diagnosis.

---

## Main Functionalities

- Binary classification of skin lesions (melanoma vs. naevus)
- Training/fine-tuning and evaluation of a EfficientNet model
- Explanation of predictions via GradCAM, DeepLIFT, and Integrated Gradients
- Visual and quantitative comparison of the explanation heatmaps
- Similarity metrics: SSIM, Pearson Correlation, Cosine Similarity, Jaccard Similarity
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

---

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

---

## How to Run

### Train a model from scratch:

   ```bash
   python src/main.py --model_filename <your_model_filename>.pth --model_path <your_model_folder> 
   ```

#### Optional training arguments:

You can also customize the training behavior by adding the following optional arguments:

| Argument           | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `--num_epochs`     | Number of training epochs (default: `58`)                                   |
| `--batch_size`     | Batch size for training and test loaders (default: `32`)                    |
| `--learning_rate`  | Learning rate for the optimizer (default: `0.00001`)                        |
| `--use_scheduler`  | Enable learning rate scheduler (linear)                                     |

### Evaluate a pretrained model:

   ```bash
   python src/main.py --load_model --model_filename <your_model_filename>.pth --model_path <your_model_folder>
   ```

### Compare explanations from two models:

   ```bash
   python src/main.py --load_model --model_filename <model_1_filename>.pth --model_path <your_model_folder> --compare_models --second_model_filename <model_2_filename>.pth
   ```

#### Make sure the dataset is located in the complete_mednode_dataset/ directory, structured with melanoma/ and naevus/ subfolders.

---

## Pretrained Models

You can download our two pretrained models from the following Google Drive link:

🔗 [Download Pretrained Models](https://drive.google.com/drive/folders/1et9_-fvm_fqbsY4yEGtCkB_whoIahXy_?usp=sharing)

After downloading, place the model files in the directory you will specify via the `--model_path` argument.

---

## Credits

- The pre-trained base model used is [`jhoppanne/SkinCancerClassifier_smote-V0`](https://huggingface.co/jhoppanne/SkinCancerClassifier_smote-V0) from Hugging Face.
- Explainability methods were implemented using [`Captum`](https://github.com/pytorch/captum) for DeepLIFT and Integrated Gradients.
- GradCAM is implemented using the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library.

This project was developed as part of the *Trustworthy and Explainable AI* course at the **University of Groningen**.

**Authors:**
- Gabriele Rubboli Petroselli – g.rubboli.petroselli@student.rug.nl  
- Thanos Bampes - a.bampes@student.rug.nl

---

## Use of Language Models

OpenAI’s ChatGPT was used in this project for code cleaning, clarification, and style formatting. All generated content has been reviewed and validated by the authors.