# XaF-Net: XAI-as-Feature Paradigm for Leakage-Free EEG-Based ADHD Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

This repository contains the official code implementation for the paper **"XaF-Net: XAI-as-Feature Paradigm for Leakage-Free EEG-Based ADHD Diagnosis"**. 

The proposed **XaF framework** transforms passive Explainable AI (XAI) heatmaps into active, highly compressed features. By enforcing a strict Subject-Independent 5-Fold Cross-Validation protocol, this codebase  mitigates epoch-wise data leakage, ensuring clinical reproducibility.

---

## 📁 Repository Structure

* `1_data_preparation_preprocessing.py` — Raw EEG processing and 3D tensor formatting
* `2_train_finder_unet.py` — Trains and Saves Finder CNN and U-Net (Attention Distillation)
* `3_extract_dynamic_masks_xaf_features.py` — Extracts final 8x8 XaF spatial heatmaps
* `4_xaf-svm-training.py` — Standalone interpretable SVM evaluation (XaF-SVM)
* `5_reproduce_xafnet_results.py` — ZERO-LEAKAGE evaluation using saved models (Fast Reproducibility)
* `6_train_xaf_net.py` — Full training pipeline for the Dual-Stream XaF-Net
* `finder-unet/` — Saved `.keras` weights for the 5-fold Finder and U-Net models
* `xaf-net models/` — Saved `.h5` weights for the final Dual-Stream XaF-Net (5 folds)
* `train-validation loss curve/` — Training history plots across all 5 folds
* `Subject_36_dynamic_mask_vs_xaf.png` — Supplementary Information

---

##  Dataset Requirements
The models are trained and evaluated on the publicly available IEEE DataPort ADHD/Control EEG dataset.
* **Dataset Link:** [IEEE DataPort (DOI: 10.21227/rzfh-zn36)](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)

##  Environment Setup
To install the required dependencies locally or in your notebook environment, run:

> `pip install tensorflow tf-keras-vis scikit-learn numpy matplotlib seaborn pandas`

---

##  Execution Guide (How to Reproduce Results)

Due to the memory constraints of processing 4D spatio-temporal EEG tensors and generating Grad-CAM masks, the pipeline is highly modularized. 

**If running in Kaggle / Google Colab:** You can run this pipeline by creating a single notebook and pasting the code from scripts `1` through `6` into sequential cells. 
> ⚠️ **NOTE ON PATHS:** Before running any script, you must update the file paths at the top of the scripts (like `eeg_data_path`, `unet_model_path`, `weights_path`) to match your local directories.

### Step 1: Data Preparation
Run `1_data_preparation_preprocessing.py`.
* Converts the raw EEG signals into 5.0-second epochs and applies 8x8 spatial interpolation.
* Outputs the base `.npz` file required for all downstream tasks.

### Step 2: Train Attention Models (Finder & U-Net)
Run `2_train_finder_unet.py`.
* **What it does:** Performs strict subject-wise data splitting, trains the *Finder CNN*, generates dynamic target masks via Grad-CAM, and trains the auxiliary *U-Net*.
* **Note:** You must execute this for all 5 folds (by updating the `TARGET_FOLD` variable or looping) to generate the complete set of models. *(Pre-trained models are provided in the `finder-unet/` folder).*

### Step 3: Extract XaF Heatmaps
Run `3_extract_dynamic_masks_xaf_features.py`.
* **What it does:** Loads the trained `.keras` models, targets the optimal convolutional layer to avoid vanishing gradients (evaluated via Hoyer Sparsity), and distills the raw EEG into highly compressed 8x8 spatial XaF heatmaps.
* **Output:** Saves fold-specific `.npz` feature files used for the final classifiers.

### Step 4: Standalone Interpretability (XaF-SVM)
Run `4_xaf-svm-training.py`.
* **What it does:** Loads the `.npz` feature files, flattens the XaF heatmaps, and trains an SVM. Aggregates predictions to the subject-level.
* **Output:** Reproduces the baseline SVM accuracy reported in the manuscript.

### Step 5: Fast Reproducibility (XaF-Net Evaluation)
** Run `5_reproduce_xafnet_results.py`.
* **What it does:** Bypasses the heavy training phase. It loads the pre-trained weights from the `xaf-net models/` directory, evaluates them on the strictly isolated test sets, and performs majority voting.
* **Output:** Reproduces the **reported accuracy** and generates the 5-fold Subject-Level Confusion Matrix reported in the paper.

### Step 6: Train XaF-Net from Scratch (Optional)
Run `6_train_xaf_net.py`.
* **What it does:** Fuses the raw 4D EEG tensors (Stream 1: 3D-CNN/BiLSTM/Transformer) with the static XaF spatial priors (Stream 2: 2D-CNN). Trains using `ModelCheckpoint` and early stopping.

---

##  Pre-Trained Models & Supplementary Material
To facilitate immediate testing, all fold-specific models are included in this repository:
* **`finder-unet/`**: Contains the intermediate `.keras` models used for Attention Distillation.
* **`xaf-net models/`**: Contains the final `.h5` weights for the dual-stream architecture.
* **Loss Curves:** Validation loss graphs. Early Stopping and ModelCheckpoint used to prevent overfitting and divergence
