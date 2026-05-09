---
title: >-
  [Paper Note] Alligat0R: Pre-Training through Covisibility Segmentation for Relative Camera Pose Regression
description: >-
  [NeurIPS 2025][Segmentation][Covisibility Segmentation] This paper replaces CroCo's cross-view completion with covisibility segmentation as a stereo vision pre-training task, predicting per-pixel labels of "co-visible / occluded / out-of-view" for each pixel. The approach significantly outperforms CroCo in low-overlap scenarios and achieves a first-place overall success rate of 60.3% on the RUBIK benchmark.
tags:
  - NeurIPS 2025
  - Segmentation
  - Covisibility Segmentation
  - Pre-Training
  - Relative Pose Regression
  - CroCo
  - ViT
date: 2026-05-08
content_hash: 132783bbc1a198f7
---

# Alligat0R: Pre-Training through Covisibility Segmentation for Relative Camera Pose Regression

**Conference**: NeurIPS 2025
**arXiv**: [2503.07561](https://arxiv.org/abs/2503.07561)
**Code**: [https://github.com/thibautloiseau/alligat0r](https://github.com/thibautloiseau/alligat0r)
**Area**: Image Segmentation
**Keywords**: Covisibility Segmentation, Pre-Training, Relative Pose Regression, CroCo, ViT

## TL;DR
This paper replaces CroCo's cross-view completion with covisibility segmentation as a stereo vision pre-training task, predicting per-pixel labels of "co-visible / occluded / out-of-view" for each pixel. The approach significantly outperforms CroCo in low-overlap scenarios and achieves a first-place overall success rate of 60.3% on the RUBIK benchmark.

## Background & Motivation
**State of the Field**: CroCo pioneered cross-view completion as a pre-training task for 3D vision and has been adopted by foundational models such as DUSt3R and MASt3R.

**Limitations of Prior Work**:
   - Cross-view completion in CroCo is ill-posed in non-co-visible regions — occluded or out-of-view pixels cannot be reconstructed from the other view.
   - CroCo requires image pairs with at least 50% overlap, limiting the diversity of training data.
   - The model learns ambiguous reconstructions in non-co-visible regions, wasting model capacity.

**Root Cause**: Cross-view completion is meaningless in non-co-visible regions, yet low-overlap image pairs are abundant in real-world scenarios.

**Paper Goals**: Design a pre-training task that provides effective supervision in both co-visible and non-co-visible regions.

**Starting Point**: Shift from "reconstruction" to "classification" — rather than reconstructing pixel values, predict the covisibility state of each pixel.

**Core Idea**: Replace cross-view completion with three-class covisibility segmentation, providing clear training signal across all regions.

## Method

### Overall Architecture
**Input**: Two images of a scene from different viewpoints. **Pre-training**: A ViT encoder independently processes both images; a Transformer decoder fuses information via cross-attention and predicts 3-class covisibility labels per pixel. **Fine-tuning**: A pose regression head is added to predict relative translation vectors and quaternion rotations.

### Key Designs

1. **Covisibility Segmentation Pre-Training Objective**:

    - **Function**: Predict per-pixel covisibility status in the other view — co-visible, occluded, or out-of-view.
    - **Mechanism**: A ViT encoder symmetrically processes both images (without masking); the decoder performs cross-view reasoning via cross-attention; an FC layer outputs a 3-class softmax. Trained with cross-entropy loss.
    - **Design Motivation**: Correct predictions require the model to understand 3D structure, occlusion relationships, and field of view, while non-co-visible regions also receive unambiguous training signal.

2. **Symmetric Forward Pass**:

    - **Function**: Both images are processed by the same encoder without masking (unlike CroCo's asymmetric masking).
    - **Design Motivation**: More consistent with downstream tasks and computationally more efficient.

3. **Two-Stage Fine-Tuning Strategy**:

    - Phase 1: Freeze the backbone and train only the pose regression head using homoscedastic loss to balance translation and rotation.
    - Phase 2: Unfreeze the entire network and jointly optimize pose loss and covisibility segmentation loss (retaining the segmentation head).
    - **Design Motivation**: Retaining the segmentation head provides regularization by preserving pre-trained representations.

### Loss & Training
- **Pre-training**: Cross-entropy loss $L_{ce} = L_{ce1} + L_{ce2}$
- **Fine-tuning**: Homoscedastic loss automatically balances translation, rotation, and segmentation terms.

## Key Experimental Results

### Main Results
Map-free Relocalization Benchmark

| Pre-training Method | $\varepsilon_t < 0.25$m (%) | $\varepsilon_t < 0.5$m (%) | $\varepsilon_t < 5$m (%) |
|---|---|---|---|
| CroCo v2 (official) | 75.7 | 87.4 | 91.5 |
| **Alligat0R** | **87.7** | **94.9** | **95.9** |

RUBIK Benchmark: Overall success rate of 60.3% ranks first, surpassing DUSt3R (54.8%) and MASt3R (53.6%).

### Ablation Study

| Configuration | Description |
|---|---|
| w/o covisibility head (removed during fine-tuning) | Performance degrades; retaining the segmentation head provides regularization. |
| nuScenes data only | ScanNet indoor data contributes complementary improvements. |
| Phase 1 only | Not unfreezing the backbone leads to substantially lower performance. |

### Key Findings
- **Large advantage in low-overlap scenarios**: In the 20–40% overlap range, Alligat0R achieves a success rate of 61.5%.
- **Speed advantage**: Direct pose regression requires only 57ms, compared to 257ms for DUSt3R.
- **Zero-shot generalization**: Zero-shot correspondence estimation on ETH3D surpasses CroCo v2.

## Highlights & Insights
- **Pre-training paradigm shift**: Moving from "reconstruction" to "classification" elegantly resolves the ill-posed nature of CroCo in non-co-visible regions.
- **Interpretability**: Segmentation outputs provide intuitive visualization of the model's geometric understanding.
- **Regularization via retained segmentation head**: Not discarding the pre-training head during fine-tuning is a generalizable fine-tuning strategy worth wider adoption.
- **Large-scale dataset Cub3**: 5M image pairs with dense covisibility annotations.

## Limitations & Future Work
- Validation is limited to pose regression; downstream tasks such as 3D reconstruction and Gaussian splatting remain untested.
- The three-class taxonomy may be too coarse, lacking fine-grained states such as partial occlusion.
- Integration with DUSt3R/MASt3R to validate improvements in 3D reconstruction is a promising direction.

## Related Work & Insights
- **vs. CroCo/CroCo v2**: CroCo is ill-posed in non-co-visible regions; this work replaces it with segmentation, providing globally valid supervision.
- **vs. DUSt3R/MASt3R**: These CroCo-based foundation models may benefit further from Alligat0R pre-training.
- **vs. Reloc3R**: Both target pose regression, but Reloc3R predicts only translation direction, whereas this work predicts metric translation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — A clean pre-training task design that directly addresses CroCo's core limitation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple benchmarks with ablations, visualizations, and generalization tests.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and fair comparisons.
- **Value**: ⭐⭐⭐⭐ — Potentially influences the pre-training paradigm of the entire CroCo ecosystem.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[CVPR 2026\] Live Interactive Training for Video Segmentation](../../CVPR2026/segmentation/live_interactive_training_for_video_segmentation.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](../../ICCV2025/segmentation/enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](../../ICCV2025/segmentation/what_if_understanding_motion_through_sparse_interactions.md)

<!-- RELATED:END -->
