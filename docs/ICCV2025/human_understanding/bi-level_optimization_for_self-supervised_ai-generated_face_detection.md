---
title: >-
  [Paper Note] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection
description: >-
  [ICCV 2025][Human Understanding][AI-generated face detection] This paper proposes BLADES, a method that employs bi-level optimization to explicitly align self-supervised pretraining with the AI-generated face detection objective. The inner loop optimizes a visual encoder on pretext tasks including EXIF classification/ranking and face manipulation detection, while the outer loop optimizes task weights to improve performance on a proxy detection task, enabling cross-generator generalization without relying on any synthetic face data.
tags:
  - ICCV 2025
  - Human Understanding
  - AI-generated face detection
  - bi-level optimization
  - self-supervised learning
  - EXIF metadata
  - anomaly detection
date: 2026-05-08
content_hash: 9ba8efc36f276d23
---

# Bi-Level Optimization for Self-Supervised AI-Generated Face Detection

**Conference**: ICCV 2025
**arXiv**: [2507.22824](https://arxiv.org/abs/2507.22824)
**Code**: [github.com/MZMMSEC/AIGFD_BLO](https://github.com/MZMMSEC/AIGFD_BLO)
**Area**: Face Understanding / AI-Generated Content Detection
**Keywords**: AI-generated face detection, bi-level optimization, self-supervised learning, EXIF metadata, anomaly detection

## TL;DR

This paper proposes BLADES, a method that employs bi-level optimization to explicitly align self-supervised pretraining with the AI-generated face detection objective. The inner loop optimizes a visual encoder on pretext tasks including EXIF classification/ranking and face manipulation detection, while the outer loop optimizes task weights to improve performance on a proxy detection task, enabling cross-generator generalization without relying on any synthetic face data.

## Background & Motivation

Faces synthesized by modern generative models (GANs, diffusion models) have reached a photorealistic level, creating an urgent need for robust detection methods. Core limitations of existing approaches:
- **Model-dependent detectors** (supervised learning) tend to overfit generator-specific artifacts seen during training and fail to generalize to emerging generative techniques.
- **Model-agnostic detectors** (relying on physiological inconsistencies such as pupil shape and corneal highlights as handcrafted features) may miss subtle statistical differences introduced by advanced generators.
- **Self-supervised feature methods** exhibit better cross-generator performance but are still suboptimal, as their pretext tasks are not explicitly designed for the detection objective.

The key insight of BLADES is that conventional self-supervised learning pipelines do not explicitly align pretext losses with the downstream task, resulting in insufficiently specialized representations. Bi-level optimization provides a principled, mathematically grounded way to end-to-end steer self-supervised pretraining toward the downstream objective.

## Method

### Overall Architecture

The system consists of two stages: (1) bi-level optimization-driven self-supervised pretraining; and (2) detection inference with the encoder frozen. The pretraining stage adopts a joint image-text embedding architecture (CLIP-style, ResNet-50 visual encoder + GPT-2 text encoder) and is trained exclusively on real face photographs.

### Key Designs

1. **Bi-Level Optimization Formulation**:

    - Inner Loop: On training set $\mathcal{B}_{tr}$, optimize encoder parameters $\boldsymbol{\theta}$ using a weighted combination of pretext task losses:
    $\boldsymbol{\theta}^{\star} = \arg\min_{\boldsymbol{\theta}} \sum_{\boldsymbol{x} \in \mathcal{B}_{tr}} \sum_{i=1}^{K} \lambda_i \ell_i(\boldsymbol{x}; \boldsymbol{\theta})$
    - Outer Loop: On validation set $\mathcal{B}_{val}$, optimize task weights $\boldsymbol{\lambda}$ to minimize the proxy detection task loss:
    $\min_{\boldsymbol{\lambda}} \sum_{\boldsymbol{x} \in \mathcal{B}_{val}} \ell_1(\boldsymbol{x}; \boldsymbol{\theta}^{\star})$
    - $\boldsymbol{\theta}$ and $\boldsymbol{\lambda}$ are updated alternately, automatically prioritizing pretext tasks most beneficial to downstream detection.

2. **Four Pretext Task Designs**:

    - **Coarse-Grained Face Manipulation Detection (proxy main task $\ell_1$)**: Manipulated faces are generated via local flipping and global affine transformations; a fidelity loss trains the encoder to distinguish "manipulated" from "photographic" faces.
    - **Categorical EXIF Tag Classification ($\ell_i$)**: Predicts categorical EXIF tags such as white balance mode and flash settings, using a focal fidelity loss to handle long-tail distributions.
    - **Ordinal EXIF Tag Ranking ($\ell_j$)**: Discretizes numerical EXIF tags (ISO, aperture, etc.) into three levels—low/medium/high—and performs pairwise ranking via the Thurstone model.
    - **Fine-Grained Face Manipulation Detection ($\ell_m$)**: Identifies specific manipulated facial regions (eyes, mouth, nose) and outputs region-level manipulation probabilities via sigmoid activations.

3. **Detection Inference Strategies**:

    - **One-Class Anomaly Detection (BLADES-OC)**: Fits a 10-component GMM to real face features; at test time, samples whose likelihood falls below the 5th percentile threshold are classified as AI-generated.
    - **Binary Classification Detection (BLADES-BC)**: Trains a lightweight two-layer perceptron (768→1536→2), augmenting AI-generated face samples with low-likelihood real faces as pseudo-outliers.

### Loss & Training

- All tasks uniformly adopt a fidelity-based loss $\ell = 1 - \sum \sqrt{p \cdot \hat{p}}$ (Bhattacharyya coefficient).
- A focal variant is applied to categorical EXIF tasks to address class imbalance, with $\gamma=2$.
- Inner loop learning rate $\alpha=10^{-5}$ (AdamW, cosine annealing); outer loop learning rate $\beta=3 \times 10^{-4}$ (Adam).
- Pretraining for 20 epochs, batch size 48, input resolution $224 \times 224$.
- Training data: 130K face photographs with EXIF metadata from the FDF dataset.

## Key Experimental Results

### Main Results (Cross-Generator Detection Accuracy %)

| Method | StyleGAN2 | VQGAN | LDM | DDIM | SDv2.1 | Midjourney | SDXL | Avg. |
|--------|-----------|-------|-----|------|--------|------------|------|------|
| CNND | 50.61 | 99.89 | 53.07 | 56.55 | 50.51 | 51.66 | 54.49 | 58.41 |
| FatFormer | 98.91 | 98.30 | 97.82 | 95.63 | 68.88 | 88.20 | 88.08 | 89.68 |
| Zou25 | 76.88 | 74.59 | 93.83 | 93.63 | 78.62 | 91.29 | 91.71 | 86.63 |
| BLADES-OC | 76.75 | 76.78 | 93.63 | 96.05 | 80.70 | 92.79 | 94.48 | 88.01 |
| **BLADES-BC** | **94.22** | **97.24** | **96.95** | **94.33** | **74.83** | **95.19** | **93.84** | **91.86** |

### Ablation Study (Feature Separability AUC %)

| Method | StyleGAN2 | LDM | SDv2.1 | FreeDoM | SDXL | Avg. |
|--------|-----------|-----|--------|---------|------|------|
| CLIP | 33.99 | 55.49 | 90.15 | 85.39 | 93.66 | 76.13 |
| FaRL | 34.35 | 47.26 | 95.24 | 79.91 | 94.61 | 75.12 |
| EAL | 69.71 | 85.81 | 74.21 | 97.92 | 93.04 | 84.12 |
| Zou25 | 85.69 | 98.66 | 88.29 | 99.79 | 97.23 | 93.43 |
| **BLADES-OC** | **87.89** | **98.24** | **91.72** | **99.92** | **98.36** | **95.05** |

### Sensitivity Analysis

| Method | TNR↑ | FPR↓ | TPR↑ | FNR↓ | F-score↑ |
|--------|------|------|------|------|----------|
| LGrad | 99.98 | 0.02 | 44.97 | 55.03 | 0.60 |
| FatFormer | 97.61 | 2.38 | 81.74 | 18.26 | 0.87 |
| **BLADES-BC** | **94.64** | **5.36** | **88.97** | **11.03** | **0.91** |

### Key Findings

- BLADES-BC achieves an average accuracy of 91.86%, substantially outperforming all competing methods.
- BLADES-OC, trained solely on real faces, surpasses most supervised methods, validating the effectiveness of the bi-level optimization alignment strategy.
- Prior methods generally exhibit high specificity (TNR > 97%) but low sensitivity (TPR < 82%); BLADES-BC achieves a balanced trade-off between the two.
- Diffusion-model-based detection methods (e.g., DIRE, AEROBLADE) also generalize poorly within the same generative family, demonstrating that reliance on model-specific cues is inherently fragile.

## Highlights & Insights

- The application of bi-level optimization is particularly elegant: the outer loop indirectly optimizes the true objective (AI-generated face detection) via a proxy task (manipulation detection), requiring no synthetic face training data whatsoever.
- The use of EXIF metadata as a self-supervised signal is novel—categorical tags are treated as classification targets while ordinal tags are used for ranking, fully exploiting the structural information in the metadata.
- The joint embedding architecture unifies heterogeneous pretext tasks within a single framework through text templates.

## Limitations & Future Work

- Pretraining relies on face photographs with EXIF metadata; a large proportion of images on social media have their EXIF information stripped.
- Detection capability for GAN-generated images under the one-class setting remains relatively limited (StyleGAN2: 76.75% only).
- The binary classification setting still requires a small amount of synthetic data to fine-tune the classification head.

## Related Work & Insights

- The idea of using bi-level optimization for task weight learning is generalizable to other multi-task self-supervised settings.
- EXIF metadata signals may be applicable to broader image provenance and integrity verification tasks.
- The fidelity loss (based on the Bhattacharyya coefficient) as an alternative to standard classification losses merits further attention.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of using bi-level optimization to align self-supervised pretraining with the detection objective is highly original, and the EXIF-based task design is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluations cover 9 generators, cross-dataset settings, feature separability, and sensitivity analysis—extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐ Addresses the generalization challenge in AI-generated content detection with significant application value for deepfake governance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)
- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[CVPR 2026\] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection](../../CVPR2026/human_understanding/regformer_transferable_relational_grounding_for_weakly-supervised_hoi_detection.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)

</div>

<!-- RELATED:END -->
