---
title: >-
  [Paper Note] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning
description: >-
  [CVPR 2026][Medical Imaging][Glioblastoma] This paper proposes RICE-NET, a multimodal 3D ResNet-18 that fuses longitudinal T1-weighted MRI with radiotherapy dose distribution maps. Evaluated on a cohort of 92 glioblastoma patients, the model achieves F1=0.916 for classifying radiation-induced contrast enhancements (RICE) versus tumor recurrence. Ablation studies reveal that the radiotherapy dose map is the single most informative modality (F1=0.78).
tags:
  - CVPR 2026
  - Medical Imaging
  - Glioblastoma
  - RICE
  - Multimodal MRI
  - Radiotherapy Dose Map
  - 3D ResNet
date: 2026-05-08
content_hash: 37ccc04cac258995
---

# Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning

**Conference**: CVPR 2026
**arXiv**: [2603.11827](https://arxiv.org/abs/2603.11827)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Glioblastoma, RICE, Multimodal MRI, Radiotherapy Dose Map, 3D ResNet

## TL;DR

This paper proposes RICE-NET, a multimodal 3D ResNet-18 that fuses longitudinal T1-weighted MRI with radiotherapy dose distribution maps. Evaluated on a cohort of 92 glioblastoma patients, the model achieves F1=0.916 for classifying radiation-induced contrast enhancements (RICE) versus tumor recurrence. Ablation studies reveal that the radiotherapy dose map is the single most informative modality (F1=0.78).

## Background & Motivation

**State of the Field**: Post-operative radiotherapy is standard of care for glioblastoma (GBM). Follow-up MRI frequently reveals new contrast-enhancing lesions that must be distinguished between tumor recurrence and radiation-induced contrast enhancement (RICE). The two conditions are highly similar in MRI appearance, and current practice relies on complex, case-by-case evaluation by multidisciplinary tumor boards.

**Limitations of Prior Work**: (1) Existing discrimination methods depend on diffusion MRI, which is not routinely acquired in clinical settings; (2) radiotherapy dose maps, increasingly discussed in tumor board meetings, have not been incorporated into automated classification models; (3) a systematic quantification of each modality's diagnostic contribution is lacking.

**Root Cause**: RICE and recurrence are difficult to distinguish on standard MRI, while radiotherapy dose maps containing critical diagnostic information remain underutilized.

**Paper Goals**: To perform automated binary classification using clinically routine T1-weighted MRI (post-operative and event scans) together with radiotherapy dose maps, and to quantify the independent contribution of each modality.

**Starting Point**: The radiotherapy dose distribution map is introduced as an additional input channel, concatenated with longitudinal MRI channels and fed into a 3D convolutional network.

**Core Idea**: The spatial distribution of radiotherapy dose is the strongest predictor for distinguishing RICE from recurrence, and its fusion with MRI further improves classification performance.

## Method

### Overall Architecture

RICE-NET is based on the 3D ResNet-18 architecture (MONAI framework). Up to three 3D volumetric inputs are concatenated along the channel dimension: (1) post-operative T1 contrast-enhanced MRI (MRI post-OP); (2) event T1-weighted MRI (MRI event); and (3) the radiotherapy dose distribution map (RD map). All volumes undergo ANTs registration, HD-BET skull stripping, and z-score normalization before being cropped to $224 \times 224 \times 224$ voxels.

### Key Designs

1. **Multimodal Channel Concatenation (Early Fusion)**: Multiple 3D volumes are directly concatenated along the channel dimension, allowing the model to learn cross-modal feature interactions through shared 3D convolutional kernels. This simplest fusion strategy is chosen to reduce overfitting risk given the small sample size (92 cases). Seven modality combinations (3 single + 3 dual + 1 full) are systematically ablated to quantify each modality's contribution.

2. **3D ResNet-18 Volumetric Encoder**: The classical 2D ResNet is extended to 3D, consisting of an initial 3D convolutional layer → 4 residual blocks (BN+ReLU) → global average pooling → fully connected classification layer. Residual connections ensure gradient propagation, while the lightweight design balances representational capacity against overfitting.

3. **Occlusion Sensitivity for Interpretability**: Small cubic 3D regions are occluded simultaneously across all registered volumes, and changes in output probability are measured to generate spatial saliency maps. Results show that model attention is highly consistent with high-dose regions and contrast-enhancing lesions, validating the clinical plausibility of the multimodal reasoning.

### Loss & Training

- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam, trained for 800 epochs
- **Validation Strategy**: 5-fold cross-validation on 80 patients; majority-vote ensemble on a held-out test set of 12 patients
- **Class Balancing**: Weighted random sampler (48 recurrence vs. 32 RICE)
- **Data Augmentation**: Elastic deformation, rotation, scaling, Gaussian noise, brightness/gamma adjustment
- **Primary Metric**: Macro F1-score

## Key Experimental Results

### Main Results

| Input Modality Combination | Val F1 | Test F1 |
|---|---|---|
| MRI event (single modality) | 0.58 | — |
| MRI post-OP (single modality) | 0.70 | — |
| RD map (single modality) | 0.78 | — |
| MRI post-OP + RD | 0.828 | — |
| MRI event + RD | 0.83 | — |
| **All three modalities** | 0.804 | **0.916** |

### Ablation Study

| Ablation | Val F1 | Analysis |
|---|---|---|
| RD map only (best single modality) | 0.78 | Dose map has the strongest predictive power |
| MRI event only (weakest single modality) | 0.58 | RICE and recurrence appear highly similar on MRI |
| MRI-only vs. with RD | Substantial gap | Dose map is indispensable |
| MRI-only val vs. test | Gap ~0.35 | Statistical uncertainty due to small sample size |

### Key Findings

- The radiotherapy dose map is the strongest single modality (F1=0.78), substantially outperforming the event MRI (0.58), indicating that the spatial dose distribution is highly predictive of tissue response.
- Full multimodal ensemble achieves test F1=0.916, demonstrating significant complementarity across modalities.
- Occlusion maps are highly correlated with high-dose regions, showing that the model simultaneously attends to dose hotspots and contrast-enhancing lesions.
- The large validation-to-test gap (~0.35) in MRI-only experiments highlights the statistical limitations of the 92-patient cohort.

## Highlights & Insights

- This is the first work to systematically incorporate radiotherapy dose maps into deep learning-based RICE/recurrence classification, with seven modality combination ablations quantifying its critical contribution.
- The use of routine T1-MRI rather than scarce diffusion MRI substantially improves clinical deployability.
- Occlusion-based interpretability aligns with clinical intuition: high-dose regions are more likely to represent RICE.
- The RD map alone achieves F1=0.78, offering direct implications for clinical decision support.

## Limitations & Future Work

- **Small Sample Size**: Only 92 cases (80 training + 12 test); statistical power is limited, and multi-center large-scale validation is required.
- **Simple Fusion Strategy**: Channel concatenation may miss complex MRI–dose interactions; cross-attention mechanisms are worth exploring.
- **Absence of Healthy Controls**: No lesion-free subjects are included, making it impossible to evaluate specificity.
- **Single-Center Data**: Data are drawn solely from Heidelberg University Hospital; multi-center validation is essential for clinical translation.

## Related Work & Insights

- Compared to diffusion MRI methods: routine MRI combined with dose maps replaces scarce diffusion MRI, offering greater clinical feasibility.
- Compared to MRI-only methods: incorporating the dose map raises F1 from ≤0.70 to 0.916.
- This work may inspire the use of dose maps as key inputs in other radiotherapy-related tasks, such as radiation pneumonitis prediction.

## Rating

⭐⭐⭐ (3/5)

**Rationale**: The clinical problem is clearly defined and practically valuable, and the idea of introducing radiotherapy dose maps is novel and effective. However, the sample size of 92 cases severely limits the reliability of the conclusions, the methodological contribution (channel concatenation + ResNet-18) lacks technical novelty, and no multi-center validation is provided. This is a clinically motivated applied study with solid but limited contributions.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_glaucoma_classification.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modalityspecific_encoders_and_partially.md)
- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ov.md)
- [\[CVPR 2026\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)

<!-- RELATED:END -->
