---
title: >-
  [Paper Note] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][Hand Pose Estimation] This paper proposes UniHOPE, the first framework to unify Hand-Only Pose Estimation (HPE) and Hand-Object Pose Estimation (HOPE). It dynamically controls outputs via an object switcher, eliminates interference from irrelevant object features through grasp-aware feature fusion, and learns occlusion-invariant features using diffusion-based generative de-occlusion combined with multi-level feature enhancement.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Hand Pose Estimation"
  - "Hand-Object Interaction"
  - "Unified Framework"
  - "Occlusion-Invariant Features"
  - "Diffusion Model De-occlusion"
date: 2026-05-08
content_hash: 0897077651f46feb
---

# UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.13303](https://arxiv.org/abs/2503.13303)  
**Code**: [GitHub](https://github.com/JoyboyWang/UniHOPE_Pytorch)  
**Area**: Human Understanding  
**Keywords**: Hand Pose Estimation, Hand-Object Interaction, Unified Framework, Occlusion-Invariant Features, Diffusion Model De-occlusion

## TL;DR

This paper proposes UniHOPE, the first framework to unify Hand-Only Pose Estimation (HPE) and Hand-Object Pose Estimation (HOPE). It dynamically controls outputs via an object switcher, eliminates interference from irrelevant object features through grasp-aware feature fusion, and learns occlusion-invariant features using diffusion-based generative de-occlusion combined with multi-level feature enhancement.

## Background & Motivation

Estimating 3D hand and potential hand-held object poses from monocular images is a long-standing challenge. Existing methods are strictly categorized into two types: (1) HPE methods that only predict hand poses without considering objects; (2) HOPE methods that assume a hand-held object is always present and estimate the object pose. Neither can flexibly adapt to general scenarios containing both with-object and without-object cases.

The authors' experiments reveal a key issue: HPE methods suffer severe performance degradation in hand-object scenarios (e.g., HandOccNet J-PE degrades from 12.98 to 19.60), and HOPE methods similarly degenerate in hand-only scenarios (e.g., Keypoint Trans. degrades from 17.99 to 25.10). Even when trained on a mixture of both datasets, the performance on the original tasks still decreases, indicating that existing methods lack the unified ability to generalize across scenarios.

Core Motivation: A unified method is needed to meet (1) basic requirements—adaptively switching between both scenarios, and (2) advanced requirements—robustly estimating hand poses regardless of the presence of objects. In particular, learning occlusion-invariant features is essential to handle severe occlusions caused by hand-held objects.

## Method

### Overall Architecture

UniHOPE consists of three core modules: (1) Dynamic Hand-Object Pose Estimation—flexibly adapting to both scenarios through an object switcher and grasp-aware feature fusion; (2) Generative De-occlusion—utilizing a diffusion model to generate paired de-occluded hand images; (3) Multi-level Feature Enhancement—learning occlusion-invariant features via self-distillation to enhance robustness.

### Key Design 1: Object Switcher and Grasp-aware Feature Fusion

**Function**: To dynamically control the switch of the object pose estimation branch and prevent feature interference in hand-only scenarios.

**Mechanism**: The object switcher predicts the grasp state confidence $s$ from the object feature $\mathbf{F}_p^o$ via an MLP. Grasp labels are automatically acquired by calculating the Relative Rotation Error (RRE) and Relative Translation Error (RTE) of the object between the initial frame and the current frame. During feature fusion, the object features are weighted by the grasp confidence: $\mathbf{F}^H = \text{Concat}(\mathbf{F}_r^h, s \cdot \mathbf{F}_r^o + (1-s) \cdot \mathbf{F}_r^h)$, where the object feature is replaced by the hand feature when $s \approx 0$.

**Design Motivation**: The hand-object information interaction structure in existing HOPE methods always propagates features from the object to the hand, which harms the hand pose estimation accuracy when no object is present. Using a soft-selection mechanism via grasp confidence avoids the incoherence of hard switching while supporting end-to-end joint optimization.

### Key Design 2: Diffusion-based Generative De-occlusion

**Function**: To generate paired de-occluded hand images, providing supervision data for learning occlusion-invariant features.

**Mechanism**: ControlNet is leveraged conditioned on depth maps and hand-object masks, adaptively adjusting the control strength $\beta$ to generate high-quality de-occluded hand images. The control strength is optimized by evaluating the finger flexion in occluded regions and the consistency in visible regions. Under severe occlusion, the strength is decreased to allow the diffusion model more imagination, while under slight occlusion, the strength is increased to preserve pose consistency.

**Design Motivation**: Ideally, features of an unoccluded hand are the optimal representation for an occluded hand. However, such paired data are extremely scarce, and the generative capability of diffusion models can create realistic de-occluded images. Since manually setting a fixed control strength yields sub-optimal results, the adaptive strategy balances plausible generation in occluded areas with consistency in visible areas.

### Key Design 3: Multi-level Feature Enhancement

**Function**: To conduct knowledge distillation from de-occluded hand images to occluded hand images, learning occlusion-invariant features.

**Mechanism**: Feature enhancement is performed at three levels—(1) Image-level: blending the encoded features of the de-occluded image with the hand-region features of the occluded image using attention; (2) Token-level: aligning their token features to pull the occluded hand features toward the de-occluded feature space; (3) Output-level: minimizing the KL divergence between the predicted MANO parameter distributions of both images. This is conducted within a self-distillation framework without requiring an extra teacher model.

**Design Motivation**: Alignment at a single level is insufficient—the image level preserves low-level details, the token level aligns mid-level semantics, and the output level ensures consistency in final predictions. Joint multi-level enhancement enables more comprehensive learning of occlusion-invariant features.

### Loss & Training

The total loss includes: the grasp state classification loss $\mathcal{L}^s$ (BCE), MANO regression loss (L1 on joints/vertices/parameters), object pose loss (rotation + translation), and image-level/token-level/output-level feature enhancement losses.

## Key Experimental Results

### Main Results: Unified Setup on DexYCB Dataset

| Method | Hand-Only J-PE ↓ | Hand-Object J-PE ↓ | Object ADD-S ↓ |
|------|-----|-----|-----|
| HandOccNet (HPE) | 13.16 | 14.58 | - |
| HFL-Net (HOPE) | 13.61 | 14.77 | 29.27 |
| **UniHOPE** | **11.39** | **12.94** | **26.76** |

### Ablation Study: Contribution of Each Component

| Configuration | Hand-Only J-PE ↓ | Hand-Object J-PE ↓ |
|------|-----|-----|
| Baseline (HOPE) | 12.52 | 13.56 |
| + Object Switcher | 12.22 | 13.42 |
| + Grasp-aware Fusion | 12.05 | 13.23 |
| + Multi-level Enhancement | **11.39** | **12.94** |

### Key Findings

- UniHOPE achieves state-of-the-art (SOTA) performance on both scenarios, with Hand-Only J-PE and Hand-Object J-PE reduced by 1.77 mm and 1.83 mm respectively compared to the best HPE and HOPE methods.
- The prediction accuracy of the object switcher for grasp state reaches 97%+, proving the effectiveness of the automatic labeling strategy.
- The quality of de-occluded images generated by the diffusion model is superior to simple image inpainting methods, and the adaptive control strength achieves a 1.2 mm improvement in PJPE compared to the fixed strength.
- Among the multi-level feature enhancements, the image-level and token-level alignments contribute the most, while the output-level alignment provides complementary gains.

## Highlights & Insights

1. **Insightful Problem Definition**: This work unifies HPE and HOPE for the first time, revealing the necessity of a unified approach through detailed cross-scenario degradation analysis.
2. **Diffusion-based De-occlusion**: Generative AI is creatively leveraged to resolve the scarcity of paired data. The adaptive control strength strategy balances plausible hallucination in occluded regions with consistency in visible regions.
3. **End-to-End Unified Training**: Soft-switching via grasp confidence avoids hard model switching and supports joint optimization.

## Limitations & Future Work

- Only single-hand scenarios are processed; two-hand interactions and multi-person scenarios are not addressed.
- The preprocessing overhead of diffusion-based de-occlusion is relatively high, which increases training preparation time, though it does not affect inference.
- Object pose estimation still relies on templates (known 3D models); template-free object reconstruction remains unexplored.
- The grasp state is strictly a binary classification (presence/absence of objects); finer-grained interaction state classification could be beneficial.

## Related Work & Insights

- **HandOccNet / SimpleHand**: Strong HPE baselines that suffer from serious cross-scenario degradation, validating the necessity of a unified approach.
- **HFL-Net**: A dual-branch HOPE method whose hand-object information interaction structure is detrimental in hand-only scenarios, which inspired the grasp-aware fusion design.
- **ControlNet**: A conditional diffusion model is creatively employed for de-occlusion data generation, demonstrating the potential of generative models in pose estimation.

## Rating

⭐⭐⭐⭐ — The problem definition is clear and practical, and the technical solution covers both basic and advanced requirements. The data augmentation approach using diffusion-based de-occlusion is highly novel. The proposed method achieves SOTA performance on both scenarios, validating the feasibility of a unified approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](../../ICCV2025/human_understanding/dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)

</div>

<!-- RELATED:END -->
