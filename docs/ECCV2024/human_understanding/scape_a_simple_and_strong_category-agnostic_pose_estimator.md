---
title: >-
  [Paper Note] SCAPE: A Simple and Strong Category-Agnostic Pose Estimator
description: >-
  [ECCV 2024][Human Understanding][Category-Agnostic Pose Estimation] This work simplifies category-agnostic pose estimation (CAPE) to a pure self-attention feature matching problem, discarding explicit similarity matching and two-stage frameworks. It introduces a Global Keypoint Feature Perceiver (GKP) and a Keypoint Attention Refiner (KAR) to improve attention quality. On the MP-100 dataset under 1-shot and 5-shot settings, it outperforms the SOTA by 2.2 and 1.3 PCK respectiv…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Category-Agnostic Pose Estimation"
  - "Few-Shot Learning"
  - "Transformer"
  - "Attention Mechanism"
  - "Keypoint Correlation"
date: 2026-05-08
content_hash: 09b9f01fc5b0bbec
---

# SCAPE: A Simple and Strong Category-Agnostic Pose Estimator

**Conference**: ECCV 2024  
**arXiv**: [2407.13483](https://arxiv.org/abs/2407.13483)  
**Code**: [https://github.com/tiny-smart/SCAPE](https://github.com/tiny-smart/SCAPE)  
**Area**: Human Understanding  
**Keywords**: Category-Agnostic Pose Estimation, Few-Shot Learning, Transformer, Attention Mechanism, Keypoint Correlation

## TL;DR
This work simplifies category-agnostic pose estimation (CAPE) to a pure self-attention feature matching problem, discarding explicit similarity matching and two-stage frameworks. It introduces a Global Keypoint Feature Perceiver (GKP) and a Keypoint Attention Refiner (KAR) to improve attention quality. On the MP-100 dataset under 1-shot and 5-shot settings, it outperforms the SOTA by 2.2 and 1.3 PCK respectively, while reducing parameter count and improving inference speed.

## Background & Motivation
Category-Agnostic Pose Estimation (CAPE) is an emerging task that aims to localize keypoints for objects of arbitrary categories given a few-shot support set. Existing methods such as POMNet and CapeFormer formulate CAPE as a similarity matching problem, designing complex explicit matching modules: POMNet generates similarity maps via convolutions to indirectly predict coordinates, while CapeFormer adopts a two-stage framework to iteratively refine unreliable initial matching results.

The key challenge is that **the self-attention mechanism of Transformers is inherently a powerful similarity calculation and matching operator**, yet prior methods stack redundant explicit matching modules on top of it. The authors observe a critical phenomenon—the implicit attention maps in CapeFormer's first stage are closer to the ground truth than its explicit similarity maps, showing clearer activation and more concentrated peaks. This indicates that explicit matching is not only redundant but might also lead to overfitting and increased matching difficulty.

Key Insight: **Since attention is matching, one should focus solely on attention quality**. The proposed architecture is simplified to pure self_attention layers and an MLP regression head, and two modules are designed specifically for the two main bottlenecks of CAPE (symmetrical keypoints and occluded keypoints) to enhance attention quality.

## Method

### Overall Architecture
SCAPE consists of four modules: (1) Feature Extractor—a shared backbone extracts features of support and query images; (2) Global Keypoint Feature Perceiver (GKP)—cross-attends support keypoints to the support image to capture global context; (3) Feature Interactor—uses self-attention layers to achieve implicit matching between support keypoints and query images; and (4) MLP Regression Head—directly regresses keypoint coordinates. The entire pipeline is designed to be single-stage and end-to-end.

### Key Designs
1. **Explicit Matching Simplified to Implicit Matching**: 

    - Mechanism: Replace explicit similarity map generation with implicit self-attention similarity matching.
    - Concatenate support keypoints `F_s` and query features `F_q` and feed them into a pure Encoder (instead of a DETR-Decoder) to simultaneously update both features.
    - Output coordinate `P = MLP(F_s)`, supervised solely by L1 loss.
    - Use different Q/K projection weights for `F_s` and `F_q` (as they carry different additional encodings—keypoint identifiers vs. positional encodings).
    - Design Motivation: Experiments in CapeFormer showed that implicit attention maps already outperform explicit similarity maps; simplification actually improves accuracy.
    - Step-by-step experiments: DETR $\rightarrow$ Encoder (+0.5), +Unshare Q/K (+0.7), reaching 89.8 PCK.

2. **Global Keypoint Feature Perceiver (GKP)**: 

    - Problem Addressed: Support keypoints contain only local features (extracted via Gaussian kernels), making them unable to distinguish symmetrical keypoints (e.g., left foot vs. right foot).
    - Mechanism: Through cross-attention, let support keypoints `F_s` act as queries to cross-attend the support image, integrating global context information.
    - Effect: After fusing global information, the similarity response of the support keypoint (e.g., left foot) to its symmetrical counterpart (e.g., right foot) is significantly reduced.
    - Practical Design: Replacing the first two self-attention blocks with two layers of GKP not only reduces parameters but also accelerates model convergence.
    - Gain: +0.5 PCK (90.3).

3. **Keypoint Attention Refiner (KAR)**: 

    - Problem Addressed: Occluded keypoints cannot find their targets in attention maps, requiring leveraging correlations with other visible keypoints to reason.
    - Observation: In earlier attention layers, the model mainly establishes correlations between keypoints, and shifts to matching with the query image only in later stages.
    - In CAPE, because the categories are unseen, the attention maps for keypoint correlations exhibit noisy patterns (unlike CSPE which has well-learned systematic patterns).
    - Core Design: Attention Filter `AF(A) = MLP(ReLU(A))`, where ReLU filters out weak, unimportant connections and MLP reconstructs correlations.
    - Multi-filter Mechanism: `KAR(A) = Σ Assign_i(F_s) · AF_i(A)`, using 4 attention filters and a keypoint weight allocator.
    - The keypoint weight allocator uses a softmax-normalized linear projection to assign similar weights to keypoints of similar nature (e.g., a cat's leg and a dog's leg).
    - Refined attention: `A_refined = softmax(A + KAR(A))`.
    - Gain: +0.7 PCK (91.6).

### Loss & Training
- Supervise the 2D coordinates output by the MLP using only L1 loss, without requiring additional similarity map supervision.
- Adam optimizer, batch size 16, trained for 180 epochs (fewer than the 210 epochs of prior work).
- Learning rate of 2e-4, decayed by 0.1 at 140 and 170 epochs.
- Configuration: 2 layers of GKP + 4 layers of self-attention feature interactors.

## Key Experimental Results

### Main Results
Comparison under 1-shot and 5-shot settings on the MP-100 dataset (PCK@0.2):

| Method | Backbone | 1-shot Mean | 5-shot Mean |
|------|----------|-------------|-------------|
| POMNet | R50 | 79.70 | 80.71 |
| CapeFormer | R50 | 85.31 | 89.30 |
| **SCAPE** | **R50** | **87.55 (+2.2)** | **90.66 (+1.3)** |
| CapeFormer | ViT-B (DINOv2) | 89.11 | 91.91 |
| **SCAPE** | **ViT-B (DINOv2)** | **91.95 (+4.4)** | **93.98 (+3.4)** |

Efficiency comparison (RTX 3090, 1-shot, R50 backbone):

| Method | Attn Blocks | $\Delta$Params (M)| FPS | PCK Mean |
|------|-------------|-------------|-----|----------|
| POMNet | 6 | +1.19 | 6.80 | 79.70 |
| CapeFormer | 9 | +7.63 | 26.09 | 85.31 |
| Lite-SCAPE | 3 | +1.95 | 36.89 | 86.13 |
| SCAPE | 6 | +3.88 | 29.43 | 87.33 |

### Ablation Study
Performance gains of each component (1-shot, split1):

| Config | Interaction Form | Unshare Q/K | KAR | GKP | PCK |
|------|---------|-------------|-----|-----|-----|
| S1 | DETR | – | – | – | 88.6 |
| S2 | Encoder | – | – | – | 89.1 |
| S3 | Encoder | ✓ | – | – | 89.8 |
| S4 | Encoder | ✓ | – | ✓ | 90.8 |
| S5 | Encoder | ✓ | ✓ | – | 91.2 |
| S6 | Encoder | ✓ | ✓ | ✓ | **91.9** |

Comparison of Matching vs. Regression:

| Paradigm | Output Format | PCK |
|------|---------|-----|
| Matching | similarity map | 86.3 |
| Regression | similarity map | 88.2 |
| Regression | coordinate | **89.1** |

Validation of Multi-Attention Filter Design in KAR:

| Number of AFs | hidden-dim | PCK |
|--------|-----------|-----|
| 1 | 50 | 90.5 |
| 1 | 200 | 90.3 |
| 4 | 50 | **91.2** |

### Key Findings
- Implicit matching outperforms explicit matching, and direct coordinate regression outperforms regressing similarity maps; **simplification leads to improvement**.
- Increasing the hidden dimension of a single AF degrades performance (-0.1), while increasing the number of AFs yields a +0.7 gain, indicating that the improvement comes from multi-view attention adjustment rather than a larger parameter scale.
- SCAPE's advantages are even more pronounced under a ViT backbone with DINOv2 pre-training (+4.4 vs +2.2), demonstrating the compatibility advantage of a simplified architecture.
- Lite-SCAPE, with only 3 attention blocks and 75% fewer parameters, outperforms CapeFormer by 0.8 PCK while improving FPS by 39%.
- SCAPE achieves the best performance across all categories in cross-supercategory generalization experiments.

## Highlights & Insights
- **Simplicity-driven Design Philosophy**: Streamlining from CapeFormer's 9 attention blocks and complex two-stage framework to 6 blocks with a single-stage MLP regression actually boosts performance.
- **Insightful Observation**: The discovery that implicit attention maps perform better than explicit similarity maps is inspiring, indicating that many seemingly necessary modules are actually redundant or even counterproductive.
- **Problem-Oriented Design**: GKP and KAR target two specific bottlenecks—symmetrical keypoints and occluded keypoints, respectively—rather than introducing modules blindly.
- **Excellent Experimental Design**: Every design decision is backed by solid control experiments, maintaining a complete and logical deductive chain.
- Bridges the keypoint correlation modeling in CAPE with practices in CSPE (e.g., human pose estimation), enabling cross-domain inheritance.

## Limitations & Future Work
- The MP-100 dataset has a limited scale (only 100 categories, 20k instances); performance on larger-scale datasets remains to be validated.
- GKP and KAR are highly customized for CAPE tasks; their efficacy in transferring to other few-shot tasks is unexplored.
- When the number of keypoints varies drastically (from a few to dozens), the performance of KAR's correlation modeling might be affected.
- The impact of stronger pre-training strategies (e.g., MAE, CLIP, etc.) on CAPE has not been explored.
- Whether large language models or vision foundation models can directly address CAPE is briefly mentioned but not thoroughly investigated.

## Related Work & Insights
- Draws inspiration from the successful practices of keypoint correlation modeling in CSPE (Category-Specific Pose Estimation) and introduces it to CAPE.
- Aligns with the evolving trend in semantic correspondence: shifting from explicit matching and post-processing to implicit Transformer matching.
- The global information integration idea of GKP can be generalized to other vision tasks requiring discrimination among similar features.
- The attention refinement mechanism of KAR provides valuable references for any Transformer application that requires modeling structural relationships among elements.

## Rating
- Novelty: ⭐⭐⭐⭐ (The simplification itself is a strong contribution; GKP and KAR are reasonably designed but not fully breakthrough)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Ablations are highly meticulous, with quantitative verification for every design decision)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation, excellent visualization, and a solid deductive chain)
- Value: ⭐⭐⭐⭐ (Establishes a stronger baseline for CAPE, though the task itself has a relatively narrow scope of application)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation](../../CVPR2025/human_understanding/recurrent_feature_mining_and_keypoint_mixup_padding_for_category-agnostic_pose_e.md)
- [\[ICLR 2026\] EdgeCAPE: Edge Weight Prediction for Category-Agnostic Pose Estimation](../../ICLR2026/human_understanding/edgecape_edge_weight_prediction_for_category-agnostic_pose_estimation.md)
- [\[ICLR 2026\] GenCape: Structure-Inductive Generative Modeling for Category-Agnostic Pose Estimation](../../ICLR2026/human_understanding/gencape_structure-inductive_generative_modeling_for_category-agnostic_pose_estim.md)
- [\[ECCV 2024\] Diffusion Model is a Good Pose Estimator from 3D RF-Vision](diffusion_model_is_a_good_pose_estimator_from_3d_rf-vision.md)
- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)

</div>

<!-- RELATED:END -->
