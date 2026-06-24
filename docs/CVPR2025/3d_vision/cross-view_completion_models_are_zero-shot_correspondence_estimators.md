---
title: >-
  [Paper Note] Cross-View Completion Models are Zero-shot Correspondence Estimators
description: >-
  [CVPR 2025][3D Vision][Cross-View Completion] Reveals that the cross-attention maps in cross-view completion (CVC) models inherently learn precise dense correspondence, and proposes ZeroCo to leverage this finding in zero-shot matching and learning-based geometric matching, significantly outperforming conventional methods based on encoder/decoder features.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Cross-View Completion"
  - "Zero-shot Matching"
  - "Cross-Attention"
  - "Cost Volume"
  - "Depth Estimation"
date: 2026-05-08
content_hash: a22e1ec5fb68a013
---

# Cross-View Completion Models are Zero-shot Correspondence Estimators

**Conference**: CVPR 2025  
**arXiv**: [2412.09072](https://arxiv.org/abs/2412.09072)  
**Code**: [cvlab-kaist.github.io/ZeroCo](https://cvlab-kaist.github.io/ZeroCo/)  
**Area**: 3D Vision / Visual Correspondence  
**Keywords**: Cross-View Completion, Zero-shot Matching, Cross-Attention, Cost Volume, Depth Estimation

## TL;DR

Reveals that the cross-attention maps in cross-view completion (CVC) models inherently learn precise dense correspondence, and proposes ZeroCo to leverage this finding in zero-shot matching and learning-based geometric matching, significantly outperforming conventional methods based on encoder/decoder features.

## Background & Motivation

- Cross-view completion (CroCo/CroCo-v2) has emerged as a powerful self-supervised pre-training task, but the mechanism of its success remains unclear.
- Existing methods leveraging CVC knowledge (DUSt3R, MASt3R, CroCo-Flow) only use decoder features as downstream representations.
- Key Insight: **Cross-attention maps in CVC models capture geometric correspondence more accurately than encoder/decoder features**.
- From an analogical perspective: The learning process of CVC is highly similar to self-supervised correspondence learning (optical flow, stereo depth)—both reconstruct the target view using source view features.

## Method

### Overall Architecture

Based on the pre-trained CroCo-v2 model, the proposed method directly utilizes the attention maps of the cross-attention layers in its decoder as the cost volume, achieving zero-shot dense correspondence estimation without any training. Furthermore, a lightweight learning module (cost aggregation + upsampling) is proposed to construct learning-based matching and depth estimation models.

### Key Designs

1. **Cross-Attention Map as Cost Volume (Zero-shot ZeroCo)**:
    - Function: Directly utilizes the cross-attention map of the CVC model for zero-shot correspondence estimation.
    - Mechanism: Extracts the pre-softmax cross-attention maps $C^l(i,j) = D_t^{l,Q}(i) \cdot D_s^{l,K}(j)$ from the original and swapped input pairs respectively, averages them across layers, and fuses them as $C' = \frac{1}{L}\sum_l C^l + (\frac{1}{L}\sum_l C^l_{\text{swap}})^T$ to enforce reciprocity, finally obtaining the flow via soft-argmax.
    - Design Motivation: The cross-attention in CVC essentially learns "how to warp features from the source view to the target view", which is equivalent to learning correspondence.

2. **Register Token Correction**:
    - Function: Eliminates artifacts in the attention map caused by register tokens.
    - Mechanism: Replaces the attention values corresponding to register tokens with the minimum attention value.
    - Design Motivation: Register tokens in Transformers lead to shortcut learning, creating artifacts where the attention inappropriately concentrates on registers instead of the correct locations.

3. **Learning-based Extension (ZeroCo-flow / ZeroCo-depth)**:
    - Function: Appends lightweight heads to the frozen cross-attention maps to achieve fine-grained matching and depth estimation.
    - Mechanism: ZeroCo-flow applies cost aggregation $\mathcal{T}_c$ and upsampling $\mathcal{U}$ along the target axis on the cross-attention map to obtain high-resolution flow. ZeroCo-depth feeds the aggregated attention map into a DPT head to predict depth.
    - Design Motivation: Zero-shot cross-attention maps have coarse resolution; adding shallow learning heads addresses fine-grained estimation.

### Loss & Training

- ZeroCo Zero-shot: No training required, directly uses pre-trained CroCo-v2 weights.
- ZeroCo-flow: Trained with standard correspondence regression loss.
- ZeroCo-finetuned: Fine-tunes the cross-attention maps themselves.
- ZeroCo-depth: Trained with reprojection, appearance consistency, and smoothness losses (commonly used losses in self-supervised depth estimation).

## Key Experimental Results

### Main Results (Zero-shot Matching HPatches-240)

| Method | AEPE ↓ (I) | AEPE ↓ (V) | AEPE ↓ (Avg) |
|------|-----------|-----------|------------|
| DINOv2 (Correlation) | 18.81 | 36.60 | 28.08 |
| DIFT_SD (Correlation) | 15.89 | 40.34 | 29.06 |
| CroCo Encoder (Corr.) | 39.69 | 54.63 | 47.52 |
| CroCo Decoder (Corr.) | 32.38 | 54.84 | 44.63 |
| **ZeroCo (Cross-attn)** | **5.07** | **13.26** | **9.41** |

### Ablation Study

| Configuration | HPatches AEPE ↓ | Explanation |
|------|----------------|------|
| Forward attention only | Higher | Lacks reciprocity constraint |
| With register tokens | Artifacts | Abnormal attention concentration |
| Register removal + Bidirectional fusion | **Optimal** | Reciprocity + artifact elimination |
| Encoder feature correlation | 47.52 | Geometric information is diluted |
| Decoder feature correlation | 44.63 | Better than encoder, but far worse than cross-attention |

### Key Findings

- The AEPE of the cross-attention map in the zero-shot setting is only 9.41, which is significantly better than DINOv2 (28.08) and diffusion model features (29.06).
- On ETH3D zero-shot matching, the average AEPE decreases from the best baseline of 25.69 to 12.72.
- Correlation maps constructed with the same encoder/decoder features from CroCo perform poorly, further proving that geometric knowledge is mainly encoded in cross-attention.
- Adding extremely lightweight heads achieves competitive results in learning-based matching and multi-frame depth estimation.
- Shows robustness to dynamic objects (does not rely on epipolar geometry).

## Highlights & Insights

- The core finding is highly insightful: cross-attention $\approx$ implicitly learned cost volumes, CVC $\approx$ self-supervised correspondence learning.
- Challenges existing practices (such as DUSt3R using decoder features), proving that cross-attention is the component actually carrying geometric knowledge.
- ZeroCo features a minimalist design: only reciprocity fusion + register correction, achieving SOTA-level zero-shot matching without training.
- Clear analogy: source-to-target warp in CVC perfectly corresponds to source-to-target alignment in optical flow/stereo matching.
- Provides a new paradigm for how pre-trained CVC models should be utilized.

## Limitations & Future Work

- The resolution of cross-attention maps is limited by the ViT patch size (usually 16×16); fine-grained matching still requires an extra learning head.
- Zero-shot inference requires two forward passes: forward and swapped.
- Only validated on CroCo-v2, other CVC models (such as cross-attention in DUSt3R decoder) have not been explored.
- Direct applications of cross-attention maps in other 3D tasks (e.g., pose estimation, 3D reconstruction) could be explored.

## Related Work & Insights

- Reveals a possible success mechanism of DUSt3R/MASt3R: correspondence priors in cross-attention.
- Contrasts with diffusion model feature matching methods like DIFT; CVC is specifically designed for geometric tasks and yields better performance.
- Register token correction is related to studies on artifacts in ViTs (Darcet et al.).
- Inspiration: Pre-trained models might contain better feature representations than those conventionally used; their internal mechanisms warrant deep analysis.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reveals a neglected but highly important finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation on zero-shot, learning-based, and depth estimation tasks, with rich visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant analogy-driven narrative, clear figures and tables.
- Value: ⭐⭐⭐⭐⭐ Reshapes the paradigm of using pre-trained CVC models, with far-reaching influence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors](genpc_zero-shot_point_cloud_completion_via_3d_generative_priors.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)

</div>

<!-- RELATED:END -->
