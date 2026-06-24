---
title: >-
  [Paper Note] DefMamba: Deformable Visual State Space Model
description: >-
  [CVPR 2025][Segmentation][State Space Models] DefMamba proposes a visual state space model based on a deformable mechanism. By dynamically adjusting the scanning path (reference point offsets + scanning order offsets) through a deformable scanning strategy, it overcomes the issue of spatial structural information loss caused by fixed scanning orders in existing Visual Mamba methods, achieving SOTA performance on ImageNet classification, COCO detection…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "State Space Models"
  - "Mamba"
  - "Deformable Scanning"
  - "Image Classification"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: f71d4442520cb847
---

# DefMamba: Deformable Visual State Space Model

**Conference**: CVPR 2025  
**arXiv**: [2504.05794](https://arxiv.org/abs/2504.05794)  
**Code**: [https://github.com/leiyeliu/DefMamba](https://github.com/leiyeliu/DefMamba)  
**Area**: Image Segmentation  
**Keywords**: State Space Models, Mamba, Deformable Scanning, Image Classification, Semantic Segmentation

## TL;DR
DefMamba proposes a visual state space model based on a deformable mechanism. By dynamically adjusting the scanning path (reference point offsets + scanning order offsets) through a deformable scanning strategy, it overcomes the issue of spatial structural information loss caused by fixed scanning orders in existing Visual Mamba methods, achieving SOTA performance on ImageNet classification, COCO detection, and ADE20K segmentation.

## Background & Motivation
Current mainstream vision foundation models are primarily based on CNNs and Transformers. CNNs are limited by the local receptive fields of sliding windows, making it difficult to efficiently aggregate global information. While Transformers excel at global information aggregation through attention mechanisms, the $O(N^2)$ computational complexity of self-attention limits their efficiency. **State Space Models (SSMs)**, particularly **Mamba**, recurrently aggregate features through a hidden state matrix, reducing computational complexity to $O(N)$ and introducing content awareness via the selection mechanism (S6), thereby establishing a promising third alternative alongside CNNs and Transformers.

However, applying Mamba to vision tasks faces a core challenge: **how to map a 2D image into a 1D sequence?** Existing methods adopt various fixed strategies:
- **Raster Scanning** (ViM, VMamba): Simple row-major traversal
- **Local Scanning** (LocalVim): Scanning within local windows
- **Continuous Scanning** (PlainMamba): Maintaining spatial continuity

However, these methods share a common issue: **using fixed scanning paths causes spatially adjacent tokens to no longer be adjacent after flattening, leading to the loss of spatial structural information in the image**. QuadMamba can adaptively adjust window sizes, but the scanning within the window remains fixed; GrootV constructs a minimum spanning tree based on adjacent features but ignores global information.

**Key Challenge**: Fixed scanning orders cannot adapt to the specific content and structure of incoming images, lacking flexibility when handling diverse object shapes.

**Core Idea**: Inspired by deformable convolution, a **deformable scanning strategy** is designed to dynamically adjust two aspects simultaneously: (1) shifting reference points to more informative locations to perceive detailed changes in objects; (2) dynamically altering the scanning order to obtain a structure-aware sequence.

## Method

### Overall Architecture
DefMamba adopts a multi-scale backbone structure similar to Swin Transformer. The image first passes through a patch embedding layer to obtain an $H/4 \times W/4 \times C$ feature map. It then processes through four stages, where the spatial resolution of the feature map gradually decreases while the channel dimension increases ($H/8 \times W/8 \times 2C$, $H/16 \times W/16 \times 4C$, $H/32 \times W/32 \times 8C$). Each stage is composed of several **Deformable Mamba (DM) blocks** and downsampling layers. The DM blocks utilize a Transformer-like structure: LayerNorm + DSSM + residual connection + LayerNorm + FFN + residual connection.

### Key Designs
1. **Deformable State Space Model (DSSM)**:
    - Retains standard forward and backward scanning branches to ensure training stability (since deformable scanning introduces more spatial skips).
    - Adds an additional deformable branch, containing deformable scanning and a deformable SSM.
    - Replaces the original 1D convolution with depthwise separable convolution to capture local features.
    - Outputs from the three branches are merged to obtain the final features.

2. **Deformable Scanning**:
    - Given input features $x \in \mathbb{R}^{H \times W \times C}$, a 3-channel offset $o \in \mathbb{R}^{H \times W \times 3}$ is generated through an offset network.
    - Offset network structure: $K \times K$ depthwise separable convolution $\rightarrow$ Channel Attention (CA) $\rightarrow$ GELU $\rightarrow$ LayerNorm $\rightarrow$ $1 \times 1$ convolution.
    - Uses $\tanh$ to constrain the offset range, preventing extreme values.
    - The 3 channels are split into: a 2-channel point offset $\Delta p$ (spatial location offset) + a 1-channel token index offset $\Delta t$ (scanning order offset).
    - Point offsets are restricted within the range of a single token (divided by H and W) to constrain the relationship between the deformable points and the reference points.

3. **Deformable Points**:
    - Generates uniform reference points $p \in \mathbb{R}^{H \times W \times 2}$, normalized to [-1, 1].
    - Deformable points: $\hat{p} = p + \Delta p$.
    - Extracts features at the deformable points on the original feature map using bilinear interpolation.
    - Designs Offset Bias (OB): Inspired by the relative position encodings in Swin Transformer, a learnable offset bias matrix $R$ is established, and position compensation is obtained via interpolation to address the issue of position encoding failure caused by point offsets.
    - Final features: $\hat{x} = \phi(x, \hat{p}) + \phi(R, \hat{p})$.

4. **Deformable Tokens**:
    - Generates reference token indices $t_r \in \mathbb{R}^{N \times 1}$, normalized to [-1, 1].
    - Deformable token indices: $t_d = t_r + \Delta t$.
    - Sorts $t_d$ to determine the new scanning order (the sorting algorithm truncates gradients; this is approximately solved using mean gradients).
    - Rearranges the offset features according to the new order to obtain a content-adaptive sequence.

5. **Offset Constraint Design Principles**:
    - The offset range of deformable points is restricted within a single token to prevent multiple deformable points from interfering with each other.
    - Channel Attention resolves the issue where depthwise convolutions cannot globally perceive token arrangements.
    - The kernel sizes for the four stages are set to [9, 7, 5, 3] to adapt to different scales.

### Loss & Training
- Classification: Standard cross-entropy with label smoothing, mixup, autoaugment, random erasing, and other augmentations.
- Detection/Segmentation: Initialized with pre-trained weights, trained using standard recipes (Mask R-CNN / UperNet).
- Optimizer: AdamW with a cosine annealing learning rate scheduler, 300 epochs of training + 20 epochs of warm-up.
- Uses EMA to stabilize training.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | DefMamba-S | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| ImageNet-1K Class. | Top-1 Acc | 83.5 | 83.4 (GrootV-T) | +0.1 |
| ImageNet-1K Class. (B) | Top-1 Acc | 84.2 | 84.2 (GrootV-S) | +0.0 |
| COCO Detection (Mask R-CNN) | APb | 47.5 | 47.4 (VMamba-T) | +0.1 |
| COCO Instance Seg. | APm | 42.8 | 42.7 (VMamba-T/GrootV-T) | +0.1 |
| ADE20K Semantic Seg. (SS) | mIoU | 48.8 | 48.5 (GrootV-T) | +0.3 |
| ADE20K Semantic Seg. (MS) | mIoU | 49.6 | 49.4 (GrootV-T) | +0.2 |

| Model | Type | Params | FLOPs | Top-1 |
|------|------|--------|-------|-------|
| DefMamba-T | SSM | 8M | 1.2G | 78.6 |
| DefMamba-S | SSM | 32M | 4.8G | 83.5 |
| DefMamba-B | SSM | 51M | 8.5G | 84.2 |
| ViM-T | SSM | 7M | 1.5G | 76.1 |
| VMamba-T | SSM | 22M | 5.6G | 82.2 |
| Swin-T | Transformer | 29M | 4.5G | 81.3 |
| ConvNeXt-T | CNN | 29M | 4.5G | 82.1 |

### Ablation Study

| Configuration | Top-1 Acc | Description |
|------|---------|------|
| FB-BB only (Forward + Backward branches) | 76.9 | Baseline |
| DB only (Deformable branch only) | 76.5 | Unstable when used alone |
| FB-BB + DB | 78.6 (+1.7) | Core gain of the deformable branch |
| FB-BB + Continuous scanning | 77.3 | Comparison with fixed scanning method |
| FB-BB + Local scanning | 77.1 | Comparison with fixed scanning method |

| Component Ablation | Top-1 | Description |
|---------|-------|------|
| Deformable branch baseline (without DP/DT) | 77.0 | |
| + DP Only (Deformable Points) | 77.4 (+0.4) | Reference point offset is effective |
| + DT Only (Deformable Tokens) | 77.2 (+0.2) | Scanning order offset is effective |
| + DP + DT | 77.9 (+0.9) | Both are complementary |
| + DP + DT + OB (Offset Bias) | 78.2 (+1.2) | Position encoding compensation is crucial |
| + DP + DT + OB + CA | 78.6 (+1.6) | Channel Attention further improves performance |

### Key Findings
- Using only the deformable branch leads to performance degradation (76.5 vs 76.9) due to excessive spatial jumping making training unstable; thus, the forward + backward branches must be retained.
- The deformable branch brings larger gains compared to continuous scanning (+0.7) and local scanning (+0.5), validating the superiority of dynamic scanning.
- Deformable points (DP) and deformable tokens (DT) are complementary: each contributes about 0.2–0.4 individually, and combining them yields a 0.9 improvement.
- Offset Bias (OB) is crucial for resolving the failure of position encodings.
- Channel Attention (CA) compensates for the limitation where depthwise convolution lacks global perception.

## Highlights & Insights
1. **Systematic design introducing the deformable mechanism into SSMs**: DefMamba does not simply transplant deformable convolutions onto Mamba; rather, it designs customized schemes such as offset constraints, index offset sorting, and offset biases specifically tailored to the characteristics of SSMs.
2. **Dual-deformable design**: Simultaneously adjusting "where to look" (point offsets) and "in what order to look" (index offsets) allows the former to capture detailed object changes and the latter to construct structure-aware sequences.
3. **Trade-offs for training stability**: The paper honestly indicates that using only the deformable branch can be unstable, resolving this by retaining the forward + backward branches as "anchors." This pragmatic design approach is highly valuable.
4. **Convincing visualizations**: Activation maps clearly demonstrate that deformable scanning focuses on object structures more accurately than raster scanning.

## Limitations & Future Work
- The authors point out two failure cases: (1) When the image only contains incomplete object structures, the offsets are too small, causing it to degenerate into fixed scanning; (2) When multiple objects are arranged regularly, the information difference between adjacent tokens is minimal, causing the model to slide into "lazy learning."
- The sorting algorithm truncates gradients, and approximating this via mean gradients may lack precision.
- The deformable point offset is restricted to the range of an individual token, limiting its adaptability to large-scale deformations.
- The performance gap compared to state-of-the-art Transformer methods (e.g., Conv2Former) remains apparent at large model scales.
- The additional parameters and computation from the offset network may become a burden in extreme efficiency-constrained scenarios.

## Related Work & Insights
- DAT (Deformable Attention Transformer) introduces the deformable mechanism to Transformers, and DefMamba analogously introduces it to SSMs.
- GrootV dynamically constructs scanning topologies based on minimum spanning trees but utilizes only adjacent features, whereas DefMamba obtains a more comprehensive perception via global channel attention.
- QuadMamba adaptively adjusts window sizes but still uses fixed scanning within the window; DefMamba achieves genuine dynamic scanning at the token level.
- Insights for segmentation tasks: Spatial structure-aware feature extraction is crucial for dense prediction tasks; the +0.3 mIoU improvement in segmentation brought by deformable scanning, though modest, points in the right direction.

## Rating
- Novelty: ⭐⭐⭐⭐ The deformable scanning strategy is pioneered in Visual Mamba, and the dual design of point and index offsets is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers classification, detection, and segmentation tasks with comprehensive ablation studies; however, a more detailed comparison with Transformer methods on segmentation is lacking.
- Writing Quality: ⭐⭐⭐⭐ The methodology is described in detail with rich visualizations, and the limitations are discussed honestly.
- Value: ⭐⭐⭐ The improvements are relatively small (mostly 0.1–0.3), and the Visual Mamba field is highly competitive and its future direction is still somewhat unclear. However, the concept of deformable scanning possesses lasting research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](groupmamba_efficient_group-based_visual_state_space_model.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)
- [\[ICCV 2025\] VSSD: Vision Mamba with Non-Causal State Space Duality](../../ICCV2025/segmentation/vssd_vision_mamba_with_non-causal_state_space_duality.md)

</div>

<!-- RELATED:END -->
