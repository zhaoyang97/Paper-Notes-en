---
title: >-
  [Paper Note] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement
description: >-
  [CVPR 2026][Medical Imaging][Breast ultrasound segmentation] Simple appearance descriptions (e.g., "dark oval") are used to drive Grounding DINO + SAM for training-free pseudo-label generation in breast ultrasound segmentation. A dual-teacher uncertainty-entropy weighted fusion mechanism and adaptive reverse contrastive learning further refine pseudo-label quality. With only 2.5% labeled data, the proposed method matches or surpasses the fully supervised upper bound.
tags:
  - CVPR 2026
  - Medical Imaging
  - Breast ultrasound segmentation
  - VLM pseudo-labels
  - appearance prompts
  - dual-teacher framework
  - uncertainty fusion
  - reverse contrastive learning
date: 2026-05-08
content_hash: 16f807d440486972
---

# A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement

**Conference**: CVPR 2026
**arXiv**: [2603.06167](https://arxiv.org/abs/2603.06167)
**Code**: None
**Area**: Medical Image Segmentation / Semi-Supervised Learning
**Keywords**: Breast ultrasound segmentation, VLM pseudo-labels, appearance prompts, dual-teacher framework, uncertainty fusion, reverse contrastive learning

## TL;DR

Simple appearance descriptions (e.g., "dark oval") are used to drive Grounding DINO + SAM for training-free pseudo-label generation in breast ultrasound segmentation. A dual-teacher uncertainty-entropy weighted fusion mechanism and adaptive reverse contrastive learning further refine pseudo-label quality. With only 2.5% labeled data, the proposed method matches or surpasses the fully supervised upper bound.

## Background & Motivation

**State of the Field**: Breast ultrasound (BUS) segmentation is a critical step in early breast cancer diagnosis. Fully supervised deep learning methods have achieved strong performance, but rely on large amounts of pixel-level annotations that are costly and require expert radiologists.

**Limitations of Prior Work**:

1. Semi-supervised methods suffer from under-trained teacher models under extremely sparse annotation (e.g., 2.5%), leading to poor-quality and structurally fragmented pseudo-labels.
2. Mainstream strong-weak data augmentation strategies are designed for RGB natural images and are ill-suited for grayscale, speckle-noisy ultrasound images.
3. While VLMs (Grounding DINO + SAM) can provide external pseudo-labels, medical-term prompts (e.g., "tumor," "high density") yield unstable zero-shot localization due to the absence of medical-domain semantics in VLMs.

**Root Cause**: High-quality external pseudo-labels are needed under sparse annotation, yet VLMs perform poorly when prompted with medical terminology.

**Paper Goals**: To obtain structurally consistent BUS pseudo-labels in a training-free manner and to effectively leverage them within a dual-teacher framework.

**Starting Point**: BUS lesions exhibit consistent visual characteristics—dark, oval or round regions. Replacing medical terminology with simple natural-language appearance descriptions bypasses the domain gap and enables cross-domain transfer.

**Core Idea**: Appearance descriptions rather than medical terms drive VLM-based pseudo-label generation, which are subsequently refined through dual-teacher fusion and reverse contrastive learning.

## Method

### Overall Architecture

The framework consists of two stages: (1) **APPG**—an LLM translates medical features into appearance descriptions, which drive Grounding DINO for detection and SAM for segmentation, generating pseudo-labels in a training-free manner; (2) **Pseudo-label refinement**—a frozen static teacher $T^A$ pre-trained on pseudo-labels captures coarse structural priors, while a dual-teacher semi-supervised framework refines pseudo-labels via UEWF fusion and AURCL boundary enhancement. The student model $S$ is jointly trained with supervised loss $\mathcal{L}_s$ on labeled data, unsupervised loss $\mathcal{L}_u$ on fused pseudo-labels, and contrastive loss $\mathcal{L}_c$.

### Key Designs

1. **APPG (Appearance Prompt-based Pseudo-label Generation)**

    - An LLM (GPT-5) translates general breast tumor medical features into concise appearance descriptions: "dark oval," "dark round," "dark lobulated."
    - These descriptions are fed into Grounding DINO to obtain bounding boxes $b_i^u = \text{VLM}_{\text{DINO}}(x_i^u, \text{aprmpt})$, and SAM generates segmentation masks $\hat{y}_i^0 = \text{SAM}(x_i^u, b_i^u)$.
    - The entire pipeline is training-free, leveraging appearance commonalities between natural and ultrasound images for cross-domain transfer.
    - Invalid pseudo-labels are filtered via an area threshold (foreground > 1%), retaining only structurally valid samples.

2. **UEWF (Uncertainty-Entropy Weighted Fusion)**

    - The static teacher $T^A$ (frozen after VLM pseudo-label pre-training) and the dynamic teacher $T^B$ (updated via EMA) each produce soft pseudo-labels $\hat{\mathbf{y}}_i^A$ and $\hat{\mathbf{y}}_i^B$.
    - Shannon entropy $\mathcal{H}(\hat{\mathbf{y}}(\mathbf{p})) = -\sum_c \hat{\mathbf{y}}_c(\mathbf{p}) \log(\hat{\mathbf{y}}_c(\mathbf{p}) + \epsilon)$ quantifies per-pixel uncertainty.
    - After patch-wise average pooling ($k=14$) for smoothing, confidence weights are computed as the reciprocal: $\mathbf{w}_{A,B} = \frac{1}{\mathbf{E}_{A,B}^{\text{smooth}} + \epsilon}$.
    - Weighted fusion: $\hat{\mathbf{y}}_i^F = \frac{\mathbf{w}_A \cdot \hat{\mathbf{y}}_i^A + \mathbf{w}_B \cdot \hat{\mathbf{y}}_i^B}{\mathbf{w}_A + \mathbf{w}_B + \epsilon}$

3. **AURCL (Adaptive Uncertainty-guided Reverse Contrastive Learning)**

    - For low-confidence pixels in the student model (selected via a dynamic top-K threshold $\tau_i = \max(\text{top-K}(\mathbf{C}_i, K), 0.2)$), predicted probabilities are inverted as $\tilde{\mathbf{p}}_i(u,v) = 1 - \mathbf{p}_i(u,v)$, forming a "reverse view."
    - Patch-level features are extracted from both original and reverse views; an InfoNCE contrastive loss pulls together positive pairs at the same location and pushes apart negatives at different locations.
    - This compels the network to learn more discriminative representations in ambiguous boundary regions.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_s + \lambda_u \mathcal{L}_u + \lambda_c \mathcal{L}_c$$

Both $\mathcal{L}_s$ and $\mathcal{L}_u$ use BCE + Dice; $\mathcal{L}_c$ is the AURCL contrastive loss. $\lambda_u=1$, $\lambda_c=0.5$. ResNet-34 backbone, input 224×224, Adam (momentum 0.995), ReduceLROnPlateau scheduler, batch size 8 (equal split of labeled and unlabeled), 100 epochs. No data augmentation is used.

## Key Experimental Results

### Main Results

| Dataset | Label Ratio | Dice (%) | IoU (%) | Prev. SOTA | Gain |
|---------|-------------|----------|---------|------------|------|
| BUSI | 2.5% | **72.72** | **63.20** | BCP 58.93 | +13.79 |
| BUSI | 10% | **77.40** | **67.13** | Text-semiseg 75.06 | +2.34 |
| BUSI | 20% | **78.38** | **68.60** | Text-semiseg 75.83 | +2.55 |
| UBB | 2.5% | **75.75** | **65.89** | Text-semiseg 59.76 | +15.99 |
| UBB | 10% | **75.95** | **65.70** | Text-semiseg 74.70 | +1.25 |
| UBB | 20% | **78.15** | **68.05** | Text-semiseg 75.55 | +2.60 |
| BUSI (fully supervised) | 100% | 81.68 | 73.74 | — | — |

On the UBB dataset with 2.5% labeled data, the proposed method achieves 75.75% Dice, surpassing the fully supervised U-Net trained on 100% annotations (74.81%).

### Ablation Study

| Component | Dice (%) | Increment |
|-----------|---------|-----------|
| Baseline (U-Net, 2.5%) | 50.00 | — |
| + Static teacher (VLM pseudo-label pre-training) | 67.34 | +17.34 |
| + Dual-teacher EMA | 71.20 | +3.86 |
| + UEWF | 71.89 | +0.69 |
| + Patch-wise smoothing | 72.20 | +0.31 |
| + AURCL | **72.72** | +0.52 |

VLM comparison: MediClipV2 achieves only 28.74% Dice; UniversalSeg 30.68%; Ours 72.72%.

### Key Findings

- APPG contributes the largest improvement (+17.34% Dice), providing stable external structural priors.
- Appearance-description prompts substantially outperform medical terminology and radiological attribute descriptions—"dark oval" yields more accurate detection boxes than "tumor."
- Patch-wise smoothing is more robust than pixel-wise smoothing.
- The advantage is even larger on the cross-device UBB dataset (+15.99% at 2.5%), demonstrating strong generalization.

## Highlights & Insights

- A single simple appearance description suffices to transfer VLMs across domains to arbitrary medical modalities; the paradigm is generalizable to dermoscopy, thyroid ultrasound, endoscopic polyp segmentation, and beyond.
- The method surpasses fully supervised performance with only 2.5% labeled data, offering significant practical value in extreme low-annotation scenarios.
- The idea of reverse contrastive learning focusing on uncertain regions is novel and complementary to conventional contrastive learning, which targets reliable regions.
- Using appearance descriptions as a cross-domain bridge is a transferable insight for other low-annotation medical imaging scenarios.

## Limitations & Future Work

- Simple appearance descriptions may be insufficient when lesion appearance is highly heterogeneous (e.g., infiltrative lesions with irregular morphology).
- Stronger VLMs (e.g., Grounded SAM 2) have not been explored; upgrading the VLM backbone could further improve pseudo-label quality.
- Validation is limited to binary segmentation (lesion/background); multi-class segmentation scenarios are not addressed.
- The no-augmentation strategy may limit applicability to other imaging domains.

## Related Work & Insights

- **vs. PH-Net (CVPR'24)**: Mines hard regions via patch-wise hardness but still relies on the model's own pseudo-labels, achieving only 55.13% Dice at 2.5%—far below the proposed method.
- **vs. Text-semiseg (MICCAI'25)**: Introduces text-driven multi-plane visual interaction to enhance pseudo-labels and is competitive at 10%/20%, but achieves only 56.85% at 2.5%, indicating that text guidance is inferior to the training-free appearance prompt + VLM approach under extreme label scarcity.
- **vs. CSC-PA (CVPR'25)**: Cross-sample prototype alignment improves semantic consistency but reaches only 58.78% at 2.5%.
- Insight: The dual-teacher uncertainty fusion mechanism is transferable to other semi-supervised detection and segmentation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The appearance prompt-driven, training-free VLM pseudo-label generation paradigm is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, three label ratios, comprehensive ablations, and cross-modality generalization validation.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, pipeline diagrams are intuitive, and ablations are progressively structured.
- Value: ⭐⭐⭐⭐ High practical value from surpassing full supervision at 2.5% annotation; paradigm is broadly applicable.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)

<!-- RELATED:END -->
