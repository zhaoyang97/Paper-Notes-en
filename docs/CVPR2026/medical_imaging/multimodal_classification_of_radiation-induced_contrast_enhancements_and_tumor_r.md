---
title: >-
  [Paper Note] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning
description: >-
  [CVPR 2026][Medical Imaging][Brain tumor] This paper proposes RICE-NET, a multimodal 3D ResNet-18 model that integrates longitudinal MRI data with radiotherapy dose distribution maps to automatically distinguish radiation-induced contrast enhancements (RICE) from tumor recurrence following glioblastoma surgery, achieving F1=0.92 on an independent test set.
tags:
  - CVPR 2026
  - Medical Imaging
  - Brain tumor
  - radiation-induced contrast enhancement
  - multimodal classification
  - longitudinal MRI
  - radiotherapy dose map
date: 2026-05-08
content_hash: 03c98ab4507371de
---

# Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning

**Conference**: CVPR 2026
**arXiv**: [2603.11827](https://arxiv.org/abs/2603.11827)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Brain tumor, radiation-induced contrast enhancement, multimodal classification, longitudinal MRI, radiotherapy dose map

## TL;DR

This paper proposes RICE-NET, a multimodal 3D ResNet-18 model that integrates longitudinal MRI data with radiotherapy dose distribution maps to automatically distinguish radiation-induced contrast enhancements (RICE) from tumor recurrence following glioblastoma surgery, achieving F1=0.92 on an independent test set.

## Background & Motivation

### 1. State of the Field
Postoperative radiotherapy is standard treatment for glioblastoma (GBM), but radiation can damage normal brain tissue. Newly appearing contrast-enhancing lesions on follow-up imaging present a critical differential diagnosis: tumor recurrence vs. radiation-induced contrast enhancement (RICE). The two are visually similar on MRI, and current practice requires time-consuming multidisciplinary tumor board evaluation of longitudinal imaging trajectories.

### 2. Limitations of Prior Work
(1) Existing methods rely on diffusion MRI that is clinically scarce; (2) most studies do not incorporate radiotherapy dose information, despite dose maps receiving increasing attention in clinical tumor boards; (3) longitudinal imaging evolution (postoperative → event-time changes) is generally ignored.

### 3. Root Cause
Clinical differentiation of RICE from tumor recurrence is extremely difficult and heavily dependent on expert experience, yet automated methods lack the ability to model the spatial distribution of radiotherapy dose and fail to fully exploit routine T1-weighted MRI (as opposed to scarce diffusion MRI).

### 4. Starting Point
Longitudinal routine T1w MRI (postoperative + event-time) and radiotherapy dose maps are combined as multi-channel inputs to a simple yet effective 3D ResNet-18 classifier.

## Method

### Overall Architecture

RICE-NET is based on 3D ResNet-18 (MONAI framework) and takes multi-channel 3D volumes as input, with each channel corresponding to a different imaging time point or modality:
- **MRI post-OP**: Postoperative T1w contrast-enhanced MRI (baseline)
- **MRI event**: T1w contrast-enhanced MRI at the time a new lesion is detected (diagnostic moment)
- **RD map**: Spatial radiotherapy dose distribution map (3D cumulative dose)

The multimodal inputs are concatenated along the channel dimension, allowing the model to learn cross-modal interactions automatically.

### Key Designs

#### 1. **Multimodal Channel Concatenation**

**Function**: MRI from different time points and the radiotherapy dose map are provided as independent channels to a 3D convolutional network.

**Mechanism**: This is the simplest multimodal fusion strategy — early fusion — allowing 3D convolutional kernels to operate jointly across spatial and channel dimensions, automatically discovering cross-modal spatial correspondences.

**Design Motivation**: The dataset is extremely small (92 cases), making complex fusion strategies prone to overfitting. Channel concatenation with a simple architecture is a practical choice under data-scarce conditions.

#### 2. **Standardized Preprocessing Pipeline**

**Function**: Ensures all input volumes are spatially aligned and of consistent size.

**Mechanism**: Isotropic resampling → ANTs registration → HD-BET brain extraction → Z-score normalization → cropping to 224×224×224. For patients with only a single-fraction dose, the total dose is estimated by scaling with the number of fractions.

#### 3. **Systematic Ablation Design**

**Function**: Systematically evaluates the contribution of each modality through seven input combinations.

**Mechanism**: Single-modality (3 variants) + dual-modality (3 variants) + full-modality (1 variant), with network architecture and training configuration held constant and only the input channels varied.

### Loss & Training

- **Loss**: Cross-entropy loss with a weighted random sampler for class balancing
- **Optimizer**: Adam
- **Training**: 800 epochs, five-fold cross-validation (80 training cases), independent test set (12 cases)
- **Data Augmentation**: Elastic deformation, rotation, scaling, Gaussian noise, brightness/gamma adjustment
- **Evaluation Metric**: Macro F1 (harmonic mean of precision and recall)
- **Interpretability**: Occlusion sensitivity maps — all registered volumes are simultaneously occluded within small 3D cubic regions, and changes in output probability are observed

## Key Experimental Results

### Main Results

**Modality Ablation Study (Macro F1)**

| Input Combination | Val F1 | Test F1 | Notes |
|---|---|---|---|
| MRI event only | 0.58 | - | Weakest single modality |
| MRI post-OP only | 0.70 | - | Postoperative baseline has moderate predictive power |
| **RD map only** | **0.78** | - | Strongest single modality |
| MRI post-OP + MRI event | 0.70 | - | Combining two MRI inputs yields no clear gain |
| MRI post-OP + RD | 0.828 | - | Postoperative MRI + dose |
| **MRI event + RD** | **0.83** | - | Best dual-modality on validation |
| All three | 0.804 | **0.916** | Best full-modality on test set |

### Ablation Study

| Configuration | Val F1 | Test F1 | Key Observation |
|---|---|---|---|
| MRI only (no RD) | 0.58–0.70 | ~0.55 | MRI-only generalizes poorly (val→test drop ~0.35) |
| Combinations including RD | 0.78–0.83 | 0.916 | RD is the critical input, substantially improves generalization |
| Full modality vs. best dual | 0.804 vs. 0.83 | 0.916 vs. — | Three-modality is slightly below dual on validation but best on test |

### Key Findings

1. **The radiotherapy dose map is the most critical input**: It achieves the highest single-modality F1 (0.78), and every combination including RD outperforms purely MRI-based combinations.
2. **MRI event has the weakest predictive power**: F1=0.58, indicating that distinguishing RICE from recurrence based solely on current imaging is extremely difficult — precisely the clinical challenge this work addresses.
3. **Generalization gap reveals statistical uncertainty**: In MRI-only experiments, the validation-to-test F1 drop of ~0.35 reflects statistical uncertainty and potential overfitting inherent to small cohorts.
4. **Occlusion maps align with clinical reasoning**: The model attends to high-dose regions and contrast-enhancing lesions, confirming multimodal reasoning rather than reliance on a single modality.
5. **Postoperative MRI + RD > MRI event + RD**: Postoperative imaging combined with dose information may already encode early markers of RICE risk, suggesting the possibility of earlier predictive modeling.

## Highlights & Insights

1. **Clinically driven perspective**: Rather than pursuing architectural complexity, this work is the first to systematically incorporate radiotherapy dose maps into imaging classification, directly addressing tumor board needs.
2. **Systematic ablation methodology**: The complete seven-combination ablation serves as a methodological template for small-data research — isolating variables is more informative than complex models.
3. **Interpretable occlusion sensitivity**: 3D occlusion maps allow radiologists to intuitively understand the model's decision basis.
4. **High clinical practicality**: Only routine T1w MRI (widely available) and the radiotherapy treatment plan (available for all radiotherapy patients) are required, with no need for scarce diffusion MRI.

## Limitations & Future Work

1. **Extremely small dataset**: Only 92 cases (80 training + 12 test), resulting in limited statistical power and a large validation-to-test gap.
2. **Simple fusion strategy**: Channel-wise concatenation may miss complex interactions between MRI and dose maps.
3. **Absence of normal controls**: No unaffected subjects are included, preventing the model from learning a "normal baseline."
4. **Binary classification**: Only RICE vs. recurrence is considered; finer-grained categories such as pseudoprogression and mixed presentations are not addressed.
5. **Single-center validation**: All data originate from Heidelberg University Hospital; multicenter validation is needed.
6. **Temporal dynamics unexplored**: Longitudinal MRI is treated as static multi-channel input; the temporal evolution of imaging is not explicitly modeled.

## Related Work & Insights

- **Diffusion MRI → Routine T1w**: Prior methods rely on advanced imaging such as ADC/DSC; this work demonstrates that routine MRI combined with dose maps can achieve high accuracy, lowering the clinical barrier to adoption.
- **Value of radiotherapy dose maps**: RD alone is the strongest single-modality input, strongly suggesting that spatial dose distribution is a core determinant of RICE formation.
- **Pragmatic strategy for small data**: 3D ResNet-18 + data augmentation + class-balanced sampling + five-fold cross-validation + multi-model ensemble voting constitutes a standard approach for small medical datasets.

## Rating

- Novelty: ⭐⭐⭐ — Methodologically straightforward (3D ResNet + channel concatenation); the primary innovation lies in the first systematic incorporation of radiotherapy dose maps.
- Experimental Thoroughness: ⭐⭐⭐ — Ablation is systematic and comprehensive, but the dataset is extremely small (12 test cases).
- Writing Quality: ⭐⭐⭐⭐ — Clinical motivation is clearly articulated; experimental design is clean.
- Value: ⭐⭐⭐⭐ — High clinical significance; demonstrates the critical role of radiotherapy dose maps in RICE differentiation and lays the groundwork for larger-scale studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_wi.md)
- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modal.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)

</div>

<!-- RELATED:END -->
