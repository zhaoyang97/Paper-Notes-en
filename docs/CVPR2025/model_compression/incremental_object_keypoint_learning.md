---
title: >-
  [Paper Note] Incremental Object Keypoint Learning (KAMP)
description: >-
  [CVPR 2025][Model Compression][Incremental keypoints] This paper defines the Incremental Keypoint Learning (IKL) paradigm for the first time—where new tasks only label new keypoints and incremental training is conducted without retaining old data. The KAMP framework is proposed to model anatomical spatial relationships between old and new keypoints using a Knowledge Association Network (KA-Net), which is combined with a keypoint-guided spatial distillation loss. Across 4 data…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Incremental keypoints"
  - "continual learning"
  - "spatial distillation"
  - "knowledge association"
  - "triangulation constraints"
date: 2026-05-08
content_hash: c0c2d01adf03fabc
---

# Incremental Object Keypoint Learning (KAMP)

**Conference**: CVPR 2025  
**arXiv**: [2503.20248](https://arxiv.org/abs/2503.20248)  
**Code**: None  
**Area**: Model Compression / Incremental Learning  
**Keywords**: Incremental keypoints, continual learning, spatial distillation, knowledge association, triangulation constraints

## TL;DR

This paper defines the Incremental Keypoint Learning (IKL) paradigm for the first time—where new tasks only label new keypoints and incremental training is conducted without retaining old data. The KAMP framework is proposed to model anatomical spatial relationships between old and new keypoints using a Knowledge Association Network (KA-Net), which is combined with a keypoint-guided spatial distillation loss. Across 4 datasets, the method not only effectively prevents forgetting but also achieves positive transfer to old keypoints (e.g., MPII AAA of 79.93% vs. 75.75% for LwF).

## Background & Motivation

**Background**: Keypoint detection models are typically trained on fixed, predefined keypoint sets. When downstream tasks require new keypoints (e.g., expanding from Downs analysis to Steiner analysis in medical imaging), the only current options are to re-annotate all data or train independent models.

**Limitations of Prior Work**: (1) Re-annotating old keypoints is impractical—annotations are costly and old data may be inaccessible due to privacy concerns; (2) training multiple independent models leads to a linear increase in model count and fails to capture structural relationships between old and new keypoints, yielding sub-optimal performance; (3) unsupervised keypoint learning (UKL) and category-agnostic pose estimation (CAPE) depend heavily on the generalization of pre-trained models, limiting their transferability.

**Key Challenge**: The critical challenge of IKL is label non-co-occurrence (LNCO)—new data only contains annotations for new keypoints, while old keypoints are unannotated. Consequently, the model cannot explicitly learn the kinematic or anatomical constraints between old and new keypoints in the label space.

**Key Insight**: There exist inherent spatial relationships between keypoints (such as the triangulation constraints between the elbow and wrist joints). An auxiliary network can be trained to infer the locations of old keypoints from new keypoints, thereby injecting this associative knowledge into the incremental learning process.

**Core Idea**: KA-Net utilizes the spatial triangulation relationship from new to old keypoints to infer old positions, which, combined with spatial softmax distillation, achieves both anti-forgetting and positive transfer of incremental keypoints.

## Method

### Overall Architecture

KAMP is a two-stage learning scheme: Stage-I (Knowledge Association) trains an auxiliary KA-Net to infer related old keypoint positions from ground-truth (GT) heatmaps of new keypoints; Stage-II (Mutual Learning) freezes the KA-Net and the old model to act as teachers, jointly training the new model to learn both old and new keypoints.

### Key Designs

1. **KA-Net (Knowledge Association Network)**: Takes the ground-truth heatmaps of two new keypoints and the visual features extracted by the frozen old model (element-wise product $\to$ concatenation $\to$ 3-layer CNN) as input, and outputs predicted heatmaps of the old keypoints. Triplets of anatomically spatially adjacent new and old keypoints (e.g., "left knee $\to$ left hip") are utilized, and training is supervised using pseudo-labels from the old model.

2. **Keypoint-guided Spatial Distillation (KSD)**: Unlike the cross-channel softmax distillation in standard LwF, spatial softmax is applied separately to each keypoint's heatmap along the H and W dimensions before computing the KL divergence: $\ell_{KSD} = \sum_j \sum_d -s^d_{sp}(\hat{y}^{t-1}_{i,j}) \cdot \log s^d_{sp}(\hat{y}^t_{i,j})$. This preserves the spatial localization information of each keypoint, making it more suitable for regression tasks than channel distillation.

3. **Auxiliary Task Creation**: Based on standard anatomical charts, the spatial proximity of new and old keypoint relationships is mapped out to select triplets (2 new + 1 old). Only one auxiliary task needs to be created per step, which can be automatically generated using GPT-4o. KA-Net is only utilized for distillation during training and is not used during inference.

### Loss & Training

$$l_{MP} = \ell_{GT} + \alpha(\ell_{KSD} + \ell_{KA})$$

- $\ell_{GT}$: $L_2$ regression loss for new keypoints.
- $\ell_{KSD}$: Spatial distillation loss (old model $\to$ new model, for all old keypoints).
- $\ell_{KA}$: Auxiliary supervision output by KA-Net (selected old keypoints).
- $\alpha$: $10^2$ (for MPII/Head-2023) or $10^4$ (for Chest/ATRW).

Based on the HRNet-W32 backbone, trained for 100 epochs with a learning rate of 2e-3 or 1e-2.

## Key Experimental Results

| Dataset | Metric | KAMP | LWF | CPR | Finetune | Joint Training |
|--------|------|------|-----|-----|----------|---------------|
| MPII 5-step | AAA₄↑ | **79.93** | 75.75 | 75.52 | 37.41 | 88.50 |
| MPII 5-step | AT₄↑ | **+1.80** | -3.86 | -3.24 | — | — |
| MPII 5-step | MT₄↑ | **+4.23** | +0.41 | +0.75 | — | — |
| Head-2023 5-step | A-MRE₄↓ | **2.32** | 4.31 | 3.71 | 51.3 | 2.12 |
| Chest 2-step | A-MRE₁↓ | **5.67** | 6.35 | 6.17 | 43.1 | 5.43 |
| ATRW 4-step | AAA₃↑ | **93.16** | 87.31 | 89.34 | 13.24 | 94.69 |

### Ablation Study

| Method | AAA₄↑ | AT₄↑ | MT₄↑ |
|------|-------|------|------|
| LWF | 75.75 | -3.86 | +0.41 |
| KAMP (only $\ell_{KSD}$) | 76.93 | -2.24 | +0.65 |
| KAMP (random KA-Net) | 77.13 | -0.48 | +1.24 |
| **KAMP (full)** | **79.93** | **+1.80** | **+4.23** |

### Comparison with Low-Shot Methods

| Method | 1-shot | 5-shot | 10-shot | 50-shot |
|------|--------|--------|---------|---------|
| CC2D | 5.14 | 4.83 | 4.08 | 3.47 |
| EGT | 5.01 | 4.58 | 3.87 | 3.21 |
| **KAMP** | **4.35** | **3.70** | **3.03** | **2.32** |

### Key Findings

- **Presence of Positive Transfer**: AT₄ = +1.80 indicates that learning new keypoints improves old keypoint detection performance on average.
- **Spatial Distillation >> Channel Distillation**: Merely changing the softmax dimension yields a 1.18% increase in AAA.
- **Anatomical Priors are Key**: Incorporating the anatomical relationship in KA-Net yields a 2.80% gain in AAA compared to randomized association.
- **IKL and CAPE are Complementary**: KAMP + MetaPoint+ achieves 79.18% PCK (vastly outperforming either method individually).
- **Annotation-Efficient**: The proposed method still outperforms dedicated few-shot methods even in 1-shot scenarios.

## Highlights & Insights

- **New Paradigm Definition**: Incremental Keypoint Learning (IKL) is defined as a natural and practical new challenge and is formally introduced for the first time.
- **Positive Transfer Beyond Anti-Forgetting**: The model not only retains old knowledge but also leverages new keypoints to enhance the detection of old keypoints.
- **Spatial Softmax Distillation**: A simple yet crucial improvement that adapts the distillation approach from classification tasks to spatial regression tasks.
- **Annotation Efficiency**: Compared to CAPE and UKL, IKL achieves better scalability with fewer annotations.

## Limitations & Future Work

- Requires manual or LLM-guided definition of anatomical associations between keypoints (triplet selection).
- Accumulated errors of pseudo-labels may worsen under long-sequence incremental settings.
- Assumes spatial correlations between old and new keypoints; when completely unrelated, the performance of KA-Net is constrained.
- Incremental learning is only explored within single object categories; cross-category incremental settings remain unexplored.
- The design of KA-Net is relatively simple (3-layer CNN); employing a stronger association network could yield further improvements.

## Related Work & Insights

- **Keypoint Estimation**: HRNet, SimpleBaseline, ViTPose—supervised learning based on a fixed keypoint set.
- **Incremental Learning**: LwF, EWC, MAS, CPR—continual learning focused on classification tasks.
- **Category-Agnostic Pose Estimation**: MetaPoint+, CAPE—relying on the generalization of pre-trained models to new keypoints.
- **Unsupervised Keypoints**: Autolink—constrained by rigid motions and video data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ An organic integration of the new paradigm, KA-Net, and spatial distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across three scenarios (medical, human body, and animals) + low-shot settings + CAPE integration.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation with an in-depth articulation of the core problem.
- Value: ⭐⭐⭐⭐⭐ Provides a practical solution for incremental annotation scenarios, with insightful findings on positive transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)
- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
