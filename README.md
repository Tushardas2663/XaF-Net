# XaF-Net: XAI-as-Feature Paradigm for Leakage-Free EEG-Based ADHD Diagnosis

This repository provides modular code and experimental results associated with the paper:

**“XaF-Net: XAI-as-Feature Paradigm for Leakage-Free EEG-Based ADHD Diagnosis”**  


The work introduces **XAI-as-Feature (XaF)**, a paradigm in which model explanation heatmaps are treated as discriminative input features for EEG-based diagnosis, instead of merely post-hoc visualizations.

---

## Repository Contents

### 1. Deep Learning Model Definitions

- **`XaFNet.py`**  
  Defines the proposed **XaF-Net** dual-stream architecture:
  - Stream I: Deep spatio-temporal EEG branch (3D CNN + BiLSTM + Transformer)
  - Stream II: XAI-embedded branch (2D CNN over XaF heatmaps)
  - Late fusion via a fully connected classification head  
  All architectural details and training hyperparameters used in the paper are documented within the file-level docstring.

- **`unet.py`**  
  Defines the auxiliary U-Net–based model used to generate dynamic attention-guided explanation masks from EEG topographical representations.  
  This model is used to produce the **XaF heatmaps** employed both as standalone features and as inputs to XaF-Net.

- **`finder.py`**  
  Contains the attention-guided mask generation finder model details used to dynamically identify salient EEG regions during self-supervised training of the auxiliary segmentation network.

Each model file includes  documentation describing the corresponding training protocol, optimization settings, and architectural configuration used in the experiments.

---

### 2. Classical Machine Learning Baselines on XaF Features

- **`ML_Models_XaF.py`**  
  Contains the dictionary and configuration of classical machine learning classifiers (SVM, XGBoost, Random Forest) used to evaluate the standalone discriminative capability of XaF heatmaps.

- **`xaf_ML_results.csv`**  
  Reports subject-wise, leakage-free 5-fold cross-validation results (mean ± standard deviation) for the above classifiers when trained exclusively on XaF features.

---

### 3. Visualization Result

- **`Subject_36_dynamic_mask_vs_xaf.png`**  
  A qualitative comparison showing:
  - Left: Average dynamic attention mask (target mask) of subject 36
  - Right: Corresponding average XaF heatmap  of subject 36

  This figure highlights the structural differences between intermediate attention masks and the final XaF representations used for classification.

---

## Experimental Protocol Summary

- Dataset: Public EEG dataset (121 children; 61 ADHD, 60 Control)
- Evaluation: Subject-wise 5-fold cross-validation (leakage-free)
- Framework: TensorFlow / Keras


Full experimental details are provided in the manuscript and in the model docstrings.

---


---

## Contact

**Tushar Das**  
National Institute of Technology Jamshedpur  
Email: 2024ugcs088@nitjsr.ac.in
