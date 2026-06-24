---
title: >-
  [Paper Note] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding
description: >-
  [CVPR 2025][3D Vision][3D Visual Grounding] This paper proposes TSP3D, the first single-stage 3D visual grounding framework based on a multi-layer sparse convolutional architecture. By achieving efficient 3D-text interaction through Text-Guided Pruning (TGP) and Completion-Based Addition (CBA), it achieves a SOTA accuracy of 46.71% Acc@0.5 at a speed of 12.43 FPS on ScanRefer.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Visual Grounding"
  - "Sparse Convolution"
  - "Text-Guided Pruning"
  - "Multimodal Fusion"
  - "Real-Time Inference"
date: 2026-05-08
content_hash: 746666382a402092
---

# Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding

**Conference**: CVPR 2025  
**arXiv**: [2502.10392](https://arxiv.org/abs/2502.10392)  
**Code**: [https://github.com/GWxuan/TSP3D](https://github.com/GWxuan/TSP3D)  
**Area**: 3D Vision  
**Keywords**: 3D Visual Grounding, Sparse Convolution, Text-Guided Pruning, Multimodal Fusion, Real-Time Inference

## TL;DR

This paper proposes TSP3D, the first single-stage 3D visual grounding framework based on a multi-layer sparse convolutional architecture. By achieving efficient 3D-text interaction through Text-Guided Pruning (TGP) and Completion-Based Addition (CBA), it achieves a SOTA accuracy of 46.71% Acc@0.5 at a speed of 12.43 FPS on ScanRefer.

## Background & Motivation

The 3D visual grounding (3DVG) task requires localizing target objects in a 3D scene based on natural language descriptions, serving as a fundamental multimodal perception task in fields such as robotics and AR/VR. Existing methods face two main challenges:

1. **Two-stage methods are slow**: The detect-then-match paradigm leads to redundant feature computation, with inference speeds typically below 3 FPS, which struggles to meet real-time requirements.
2. **Single-stage methods lack precision**: Point cloud-based architectures (e.g., PointNet++) rely on time-consuming operations like FPS and kNN, and require aggressive downsampling, which discards fine-grained geometric details and keeps speed below 6 FPS.

The authors observe that multi-layer sparse convolutional architectures have already achieved leading speed and accuracy in 3D object detection. However, directly transferring this to 3DVG encounters a core challenge: 3DVG requires deep interaction between 3D scene representations and text features, but the massive number of voxels in sparse convolutional architectures (which grows exponentially with upsampling) makes the cross-modal attention computational cost prohibitive.

Therefore, how to achieve deep multimodal interaction while preserving the efficiency of sparse convolution is the core problem addressed in this paper.

## Method

### Overall Architecture

TSP3D builds a single-stage 3DVG pipeline based on a three-layer sparse convolutional architecture. The input point cloud is voxelized and passed through three MinkResBlocks to extract multi-layer voxel features $V_l$ ($l=1,2,3$), while text features $T$ are encoded using RoBERTa. The core innovation lies in introducing the TGP and CBA modules during the upsampling stage, to merge 3D and text features layer by layer and predict target bounding boxes via convolutional heads.

### Key Designs

**Design 1: Text-Guided Pruning (TGP)**

- **Function**: Achieve deep cross-modal interaction while maintaining efficient inference.
- **Mechanism**: Before upsampling at each layer, text information is utilized to guide voxel pruning, progressively focusing on the target region. Two levels of TGP are configured: scene-level TGP (level 3 $\rightarrow$ 2) to distinguish foreground/background, and object-level TGP (level 2 $\rightarrow$ 1) to focus on the targets and referents mentioned in the text.
- **Design Motivation**: Due to the massive number of voxels in sparse convolutions, direct cross-attention computation is infeasible. By pruning prior to interaction, the voxel count is reduced to approximately 7% of the original, making attention calculation practical.

Specifically, TGP performs FPS sampling on voxel features, performs cross-attention with text features, and predicts the retention probability $\hat{M}$. A pruning mask is obtained via binarization using the Heaviside step function:

$$U_l^P = U_l \odot \Theta(\mathcal{I}(\hat{M}, U_l) - \sigma)$$

A simplified version removes FPS and merges feature enhancement with pruning prediction into a single interaction executed prior to pruning, further reducing computational overhead.

**Design 2: Completion-Based Addition (CBA)**

- **Function**: Restore the geometric information of target regions that were mistakenly removed during the pruning process.
- **Mechanism**: Use text features to query voxels in the backbone features that might belong to the target, complementing the pruned upsampled features. First, cross-attention is used to enhance backbone features $V_l'$, followed by predicting a target region mask $M_l^{tar}$ to identify mistakenly pruned positions $M_l^{mis}$, which are then completed using interpolation.
- **Design Motivation**: Pruning inevitably discards representations of small or narrow objects. While full addition is computationally expensive and compromises focus, purely pruning-aware addition relies too heavily on pruning results. CBA precisely restores key regions with minimal computational cost.

$$M_l^{mis} = M_l^{tar} \wedge (\neg \mathcal{C}(U_l^G, V_l))$$

**Design 3: Multi-Level Supervision Strategy**

- **Function**: Provide precise training signals for TGP and CBA.
- **Mechanism**: Scene-level TGP supervision generates an $L \times L \times L$ cubic mask based on all object centers; object-level TGP generates masks based on the target and referents mentioned in the text; CBA completion supervision is shared with classification, using voxels near the target object center as positive samples.
- **Design Motivation**: Different levels have different pruning targets: the scene-level needs to retain all object regions, while the object-level must focus on semantically relevant regions.

### Loss & Training

The total loss is the sum of four parts:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{pruning} + \lambda_2 \mathcal{L}_{com} + \lambda_3 \mathcal{L}_{class} + \lambda_4 \mathcal{L}_{bbox}$$

where pruning, completion, and classification losses utilize Focal Loss to address class imbalance, and bounding box regression utilizes DIoU loss. All weights $\lambda$ are set to 1.

## Key Experimental Results

### Main Results: ScanRefer Dataset

| Method | Type | Acc@0.25 | Acc@0.5 | FPS |
|------|------|----------|---------|-----|
| MCLN (ECCV'24) | Two-stage | 57.17 | 45.53 | 3.17 |
| G³-LQ (CVPR'24) | Two-stage | 56.90 | 45.58 | - |
| EDA (CVPR'23) | Single-stage | 53.83 | 41.70 | 5.98 |
| MCLN (ECCV'24) | Single-stage | 54.30 | 42.64 | 5.45 |
| **TSP3D (Ours)** | **Single-stage** | **56.45** | **46.71** | **12.43** |

### NR3D/SR3D Dataset

| Method | Type | NR3D | SR3D |
|------|------|------|------|
| MCLN (ECCV'24) | Two-stage (det) | 46.1 | 53.9 |
| MCLN (ECCV'24) | Single-stage | 45.7 | 53.4 |
| **TSP3D (Ours)** | **Single-stage** | **48.7** | **57.1** |

### Ablation Study

| Component | Acc@0.25 | Acc@0.5 | FPS |
|------|----------|---------|-----|
| TSP3D-B (Baseline) | 44.87 | 29.05 | 14.58 |
| + TGP | 54.63 | 44.91 | 12.84 |
| + TGP + CBA (Full TSP3D) | 56.45 | 46.71 | 12.43 |

### Key Findings

1. TSP3D is the first 3DVG method to exceed 10 FPS, speeding up inference by more than 100% compared to previous fastest single-stage methods.
2. TGP reduces the number of voxels to ~7% of the original while significantly improving accuracy (+15.86 Acc@0.5).
3. CBA brings a +1.80 Acc@0.5 improvement at a minimal speed cost (-0.41 FPS).
4. As a single-stage method, it surpasses all two-stage methods in accuracy for the first time.

## Highlights & Insights

1. **Paradigm Innovation of Sparse Convolution + Pruning**: For the first time, a multi-layer sparse convolutional architecture is introduced to 3DVG. The "pruning before interaction" strategy elegantly addresses the issue of cross-attention being infeasible due to the massive number of voxels.
2. **Progressive Focus Mechanism**: The two-level pruning design (scene-level $\rightarrow$ object-level) mimics the human visual search process of "first scanning the panorama, then focusing on details."
3. **Complementary Design of Completion and Pruning**: CBA does not simply restore all pruned regions but selectively completes regions that may belong to the target, balancing efficiency and accuracy.

## Limitations & Future Work

1. Pruning threshold $\sigma$ and completion threshold $\tau$ need to be manually adjusted; different scenes may require different settings.
2. The text encoder uses a fixed RoBERTa, without exploring the impact of stronger language models (e.g., LLMs) on performance.
3. Generalization ability has not been verified on larger-scale 3D scenes (e.g., outdoor scenes).
4. The completion module depends on the quality of backbone features; when initial feature extraction is poor, the completion effect may be limited.

## Related Work & Insights

- **FCAF3D/TR3D**: The success of multi-layer sparse convolution in 3D detection directly inspired the architectural design of TSP3D.
- **DSPDet3D**: The experience of sparse convolution in handling small objects aligns with the completion concept of CBA.
- **BUTD-DETR/EDA**: Pioneers of single-stage 3DVG, but limited by the efficiency bottlenecks of point cloud architectures.
- **Insight**: In other tasks requiring large-scale feature interactions, the "pruning before interaction" strategy could be equally effective.

## Rating

⭐⭐⭐⭐ — Achieves real-time inference (12.43 FPS) in 3DVG for the first time, with accuracy comprehensively surpassing existing methods. The design of "Text-Guided Pruning" is simple yet highly efficient, serving as an excellent paradigm for extending sparse convolution to multimodal tasks. The limitation lies in experiments being restricted to indoor datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding](../../CVPR2026/3d_vision/pv-ground_text-guided_point-voxel_interaction_for_3d_visual_grounding.md)
- [\[CVPR 2025\] ProxyTransformation: Preshaping Point Cloud Manifold with Proxy Attention for 3D Visual Grounding](proxytransformation_preshaping_point_cloud_manifold_with_proxy_attention_for_3d_.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [\[CVPR 2025\] ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion](shapeshifter_3d_variations_using_multiscale_and_sparse_point-voxel_diffusion.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)

</div>

<!-- RELATED:END -->
