---
title: >-
  [Paper Note] Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging
description: >-
  [CVPR2025][Medical Imaging][Non-invasive blood glucose estimation] Proposes ScleraGluNet, which utilizes five-direction scleral blood vessel images integrated with a multi-branch CNN, MRFO feature selection, and Transformer cross-view fusion, achieving a three-class metabolic state classification accuracy of 93.8% and continuous fasting plasma glucose estimation with an MAE of 6.42 mg/dL, offering a novel approach for non-invasive blood glucose monitoring.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Non-invasive blood glucose estimation"
  - "scleral blood vessel imaging"
  - "multi-view learning"
  - "Transformer fusion"
  - "MRFO feature optimization"
date: 2026-05-08
content_hash: 4da726291b785ab7
---

# Deep Learning Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging

**Conference**: CVPR2025  
**arXiv**: [2603.12715](https://arxiv.org/abs/2603.12715)  
**Code**: Not open-sourced  
**Area**: Medical Imaging  
**Keywords**: Non-invasive blood glucose estimation, scleral blood vessel imaging, multi-view learning, Transformer fusion, MRFO feature optimization

## TL;DR

Proposes ScleraGluNet, which utilizes five-direction scleral blood vessel images integrated with a multi-branch CNN, MRFO feature selection, and Transformer cross-view fusion, achieving a three-class metabolic state classification accuracy of 93.8% and continuous fasting plasma glucose estimation with an MAE of 6.42 mg/dL, offering a novel approach for non-invasive blood glucose monitoring.

## Background & Motivation

- The global prevalence of diabetes reached 537 million in 2021 and is projected to reach 783 million by 2045, with chronic hyperglycemia leading to microvascular and macrovascular complications.
- Standard diagnostic tests (FPG, OGTT, HbA1c) require invasive blood sampling, posing a heavy burden for daily monitoring; although CGM reduces finger-prick tests, it still requires subcutaneous implantation and remains costly.
- Ocular surface microvessels (sclera/conjunctiva) are directly observable, and existing studies have confirmed that diabetes induces morphological changes in conjunctival vessels (e.g., tortuosity, density changes); however, systematic multi-view exploitation based on deep learning is lacking.
- **Core Motivation**: Single-view acquisition loses heterogeneous vascular information across different scleral regions; multidirectional acquisition and cross-view fusion are necessary to comprehensively capture blood glucose-related microvascular features.

## Method

### Overall Architecture
ScleraGluNet is a multi-view multi-task deep learning architecture consisting of four core modules:

- Image preprocessing and blood vessel enhancement
- Five-pathway parallel CNN feature extraction
- MRFO feature refinement + Transformer cross-view fusion
- Dual classification/regression outputs

### Data Collection and Preprocessing
- **Dataset**: 445 subjects (150 normal / 140 controlled diabetes / 155 hyperglycemic diabetes), with five gaze directions per subject (primary, superior, inferior, nasal, temporal), totaling 2,225 anterior segment images.
- **Preprocessing Pipeline**: Quality control $\rightarrow$ ROI extraction (removing eyelid/eyelash background) $\rightarrow$ Color/brightness normalization $\rightarrow$ CLAHE contrast enhancement $\rightarrow$ Frangi filter tubular structure enhancement $\rightarrow$ Binary mask validation.

### Network Design
1. **Five-Pathway Parallel CNN Branches**: Each gaze direction corresponds to a CNN branch with independent parameters, extracting direction-specific local vascular features (caliber changes, tortuosity, branching complexity).
2. **MRFO Feature Refinement**: The Manta Ray Foraging Optimization (MRFO) algorithm selects a feature subset, removing redundant or highly correlated features and retaining the most discriminative vascular representations.
3. **Transformer Cross-View Fusion**: A self-attention mechanism models long-range dependencies across different scleral regions, identifying cross-quadrant vascular patterns (e.g., temporal-nasal asymmetric remodeling).
4. **Dual Output Heads**: A classification head outputs probabilities for three metabolic states, while a regression head estimates continuous FPG (mg/dL).

### Loss & Training
Composite loss = Cross-entropy loss (classification) + MSE loss (regression), optimized via multi-task joint learning.

### Training Strategy
- Subject-level five-fold cross-validation (GroupKFold) is utilized, where all images from a single subject appear only in the same fold to prevent data leakage.
- 95% CIs are estimated via subject-level bootstrap resampling (1000 iterations).
- The Adam optimizer is employed, with the learning rate, batch size, epoch number, and task loss weights tuned on the validation set.
- Evaluation is performed at the subject level (rather than the image level) to ensure the results reflect true generalization capability.

## Key Experimental Results

| Metric | Value |
|------|------|
| Three-class overall accuracy | 93.8% (five-fold mean 93.7% $\pm$ 0.7%) |
| Normal group recall | 94.0% (141/150) |
| Controlled diabetes recall | 92.1% (129/140) |
| Hyperglycemic diabetes recall | 93.5% (145/155) |
| AUC (Normal / Controlled / Hyperglycemic) | 0.971 / 0.956 / 0.982 |
| FPG estimation MAE | 6.42 mg/dL |
| FPG estimation RMSE | 7.91 mg/dL |
| Pearson r / $R^2$ | 0.983 / 0.966 |
| Bland-Altman mean bias | +1.45 mg/dL |
| 95% limits of agreement | -8.33 to +11.23 mg/dL |

**Ablation Study** (incremental classification accuracy):
- Single-view CNN baseline $<$ Multi-view CNN (without MRFO/Transformer) $<$ Multi-view + MRFO $<$ Full ScleraGluNet, demonstrating significant contributions from each module.

**Key Findings**:
- Misclassifications primarily occur between adjacent metabolic categories (controlled vs. hyperglycemic), aligning with the clinical characteristics of the blood glucose continuum.
- Grad-CAM/Grad-CAM++ visualizations reveal that the model focus is concentrated on the scleral vessel regions, with the hyperglycemic group displaying consistent strong activation across different gaze directions.
- Five-fold accuracy is highly stable: individual fold accuracies range from 92.8% to 94.6% with a standard deviation of only 0.7%, indicating that the results do not rely on favorable data partitioning.
- Representative case analysis: vessels in the normal group are thin and uniform; the controlled group exhibits mild tortuosity; and the hyperglycemic group shows prominent vasodilation, spiral structures, and uneven caliber changes.

## Highlights & Insights

- **Innovative Multidirectional Acquisition Design**: Systematically utilizes scleral images from five gaze directions for the first time, capturing spatially heterogeneous microvascular information.
- **Complete Closed Loop**: An end-to-end design spanning from the image acquisition protocol to preprocessing, feature extraction, fusion, and dual-task output.
- **Clinical Feasibility**: Requires only an anterior segment camera (no dilation or fundus imaging needed), making it highly suitable for telemedicine and large-scale screening.
- **Rigorous Validation**: Incorporates subject-level splitting, bootstrap CIs, and Bland-Altman analysis, avoiding common data leakage issues.
- **Dual-Task Joint Learning**: Classification and regression tasks share feature representations, mutually enhancing performance.

## Limitations & Future Work

- Single-center study (Changsha Aier Eye Hospital), lacking multi-center external validation, leaving its generalizability to be verified.
- Confounding factors that may affect scleral vessels, such as hypertension, smoking, and anemia, were not controlled.
- Focuses solely on fasting plasma glucose without incorporating postprandial blood glucose or longitudinal monitoring data.
- Grad-CAM only provides coarse localization, which cannot serve as an exact indicator of vascular pathology.
- The dataset scale is limited (445 subjects), posing a risk of overfitting for the deep learning models.

## Related Work & Insights

- **Retinal Imaging**: Existing deep learning systems predict cardiometabolic states and HbA1c from retinal images, but require expensive fundus cameras.
- **Conjunctival Microcirculation**: Previous studies recorded diabetes-related conjunctival vascular alterations using OCTA and red-free imaging, but did not construct an end-to-end DL system.
- **PPG/Thermal Imaging**: Consumer-grade devices estimate blood glucose but are sensitive to motion/lighting with weak physiological coupling.
- **MRFO-INEYENET**: The authors' previous work utilized only single-angle ocular images with MRFO optimization; ScleraGluNet introduces multidirectional acquisition and Transformer fusion on top of it.
- **Association Between Scleral/Conjunctival Vessels and Metabolism**: Multiple OCTA studies have confirmed microvascular changes in the sclera of patients with diabetes, providing a physiological basis for the study.
- **Multi-View Learning**: Acquiring multi-angle data enhances model robustness and generalizability, which has been validated across various computer vision domains.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (Non-invasive blood glucose estimation combining multidirectional scleral imaging and cross-view fusion is a novel approach.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Includes five-fold cross-validation, ablation study, Bland-Altman, and Grad-CAM, but lacks external validation.)
- **Writing Quality**: ⭐⭐⭐ (Structurally clear but suffers from descriptive redundancies and incoherent paragraphs in the introduction section.)
- **Value**: ⭐⭐⭐⭐ (Holds promising clinical application prospects, but requires multi-center validation for actual deployment.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[CVPR 2025\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2025\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2025\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [\[CVPR 2025\] Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification](transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)

</div>

<!-- RELATED:END -->
