---
title: >-
  [Paper Note] DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
description: >-
  [CVPR 2025][Segmentation][RGBD segmentation] Proposes utilizing the depth map directly as a geometric prior instead of encoding it through neural networks. It designs Geometry Self-Attention (GSA) to fuse depth distance and spatial distance into decay factors that modulate attention weights, matching or surpassing dual-encoder RGBD segmentation methods with approximately half the FLOPs.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "RGBD segmentation"
  - "geometric prior"
  - "depth map"
  - "self-attention improvement"
  - "semantic segmentation"
date: 2026-05-08
content_hash: 98cadbb87e09b58b
---

# DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2504.04701](https://arxiv.org/abs/2504.04701)  
**Code**: [https://github.com/VCIP-RGBD/DFormer](https://github.com/VCIP-RGBD/DFormer)  
**Area**: Segmentation  
**Keywords**: RGBD segmentation, geometric prior, depth map, self-attention improvement, semantic segmentation

## TL;DR
Proposes utilizing the depth map directly as a geometric prior instead of encoding it through neural networks. It designs Geometry Self-Attention (GSA) to fuse depth distance and spatial distance into decay factors that modulate attention weights, matching or surpassing dual-encoder RGBD segmentation methods with approximately half the FLOPs.

## Background & Motivation

**Background**: RGBD semantic segmentation typically utilizes dual encoders to process RGB and Depth separately before fusion. The depth map is processed through a full encoder backbone, which almost doubles the parameters and computational cost.

**Limitations of Prior Work**: Dual-encoder schemes are computationally intensive (e.g., GeminiFusion requires 256G FLOPs), and the representations learned by the depth encoder may not align with the RGB encoder. Depth maps are inherently geometric information, and using neural networks to "encode" them is an indirect and wasteful way of utilization.

**Key Challenge**: Depth maps provide explicit 3D geometric relationships (inter-object distances, co-planarity), but after being encoded by neural networks, this explicit geometric information becomes implicit. As a result, the model must re-learn information that was already explicit.

**Goal**: To directly utilize the geometric information of depth maps as an attention prior, rather than encoding them into features.

**Key Insight**: Depth maps can directly indicate which patches are close in 3D space (likely belonging to the same object) and which are far apart. This geometric relationship is converted into a decay factor for self-attention—patches close in 3D space receive stronger attention, while those far apart receive weaker attention.

**Core Idea**: Use a fused prior of depth distance and spatial distance as a geometric decay factor for attention, allowing effective utilization of depth information without a depth encoder.

## Method

### Overall Architecture
RGB image $\to$ single-encoder ViT + GSA (Geometry Self-Attention) $\to$ lightweight decoder head. The depth map does not pass through an encoder; instead, it is directly utilized as a prior in each attention layer.

### Key Designs

1. **Geometry Self-Attention (GSA)**:

    - **Function**: Modulates standard self-attention using depth geometric information.
    - **Mechanism**: $$\text{GeoAttn}(Q,K,V,G) = (\text{Softmax}(QK^T) \odot \beta^G)V$$, where $G$ is the geometric prior matrix and $\beta \in (0,1)$ is a learnable decay base. A large geometric prior $\to$ $\beta^G$ approaches 0 (suppressing attention), while a small geometric prior $\to$ $\beta^G$ approaches 1 (preserving attention).
    - **Design Motivation**: Converts depth information from "features" to "attention weight modulation", requiring no additional encoder parameters.

2. **Geometric Prior Fusion**:

    - **Function**: Combines depth distance and spatial distance, two types of geometric signals.
    - **Mechanism**: $G$ fuses depth distance $D_{ij}$ (the difference in depth values between two patches) and spatial Manhattan distance $S_{ij}$ (the coordinate difference of patches in the image) via learnable memory weights. Memory-based fusion performs better than convolution fusion, addition fusion, or element-wise multiplication fusion (56.2 vs 55.8/54.6/54.9 mIoU).
    - **Design Motivation**: Depth distance reflects 3D relationships while spatial distance reflects 2D proximity, which are complementary.

3. **Axes Decomposition**:

    - **Function**: Reduces the computational complexity of GSA.
    - **Mechanism**: Decomposes the 2D geometric prior into horizontal and vertical directions for separate calculation, cutting computation in half. The performance drops slightly (56.0 vs 56.2 mIoU) but computation is significantly reduced.
    - **Design Motivation**: Standard global GSA is not scalable at high resolutions.

### Loss & Training
Standard cross-entropy segmentation loss. RGB-D pre-training is performed on ImageNet-1K (with depth maps generated by depth estimation models).

## Key Experimental Results

### Main Results

| Model | Params | FLOPs | NYU mIoU | SUN mIoU |
|------|------|-------|---------|---------|
| GeminiFusion (MiT-B5) | 137.2M | 256.1G | 57.7 | 53.3 |
| **DFormerv2-B** | **53.9M** | **67.2G** | **57.7** | 52.1 |
| **DFormerv2-L** | **95.5M** | **124.1G** | **58.4** | **53.3** |

### Ablation Study

| Configuration | NYU mIoU | Explanation |
|------|---------|------|
| Standard attention | 51.7 | No depth |
| + Depth prior | 54.3 | +2.6 |
| + depth + spatial prior | 56.2 | +4.5 |
| + Axes decomposition | 56.0 | Computation halved, only -0.2 |

### Key Findings
- **Matching dual-encoder with half the footprint**: DFormerv2-B (67G FLOPs) matches the accuracy of GeminiFusion (256G FLOPs) with only 26% of the computation.
- **Geometric prior contributes +4.5 mIoU**: Performance improves from 51.7 to 56.2, which is more effective than any other way of utilizing depth.
- **First to prove that depth maps do not require an encoder**—directly acting as an attention prior is sufficient.

## Highlights & Insights
- **The paradigm of "geometric prior replacing depth encoding"** is elegant—depth maps inherently consist of explicit geometric quantities, requiring no "re-understanding" by neural networks.
- **The design of the decay factor $\beta^G$** is simple yet effective—naturally converting distance to attention weights through exponential decay.

## Limitations & Future Work
- It heavily relies on the quality of depth maps; errors in estimated depth will propagate to the geometric prior.
- Only validated on indoor scenes (NYU/SUN), whereas depth ranges in outdoor scenes vary significantly.
- The interpretability of memory-based fusion is limited.

## Related Work & Insights
- **vs CMX / CMNext**: These are dual-encoder fusion schemes with high computational costs. DFormerv2 achieves comparable performance using a single encoder combined with a geometric prior.
- **vs DFormer v1**: v1 also uses only a single encoder but lacks an explicit geometric prior. The GSA in v2 provides a more principled way of utilizing depth.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using depth maps as attention priors instead of encoded features is a paradigm innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three datasets (NYU/SUN/Deliver) with detailed comparisons of fusion strategies.
- Writing Quality: ⭐⭐⭐⭐ The motivation for the methodology is clear.
- Value: ⭐⭐⭐⭐⭐ Holds significant importance for the field of RGBD understanding, possessing a clear efficiency advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [\[CVPR 2025\] Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation](soft_self-labeling_and_potts_relaxations_for_weakly-supervised_segmentation.md)
- [\[CVPR 2025\] Universal Domain Adaptation for Semantic Segmentation](universal_domain_adaptation_for_semantic_segmentation.md)
- [\[ECCV 2024\] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](../../ECCV2024/segmentation/sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [\[ECCV 2024\] Attention Decomposition for Cross-Domain Semantic Segmentation](../../ECCV2024/segmentation/attention_decomposition_for_cross-domain_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
