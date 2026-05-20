---
title: >-
  [Paper Note] UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale
description: >-
  [ICCV 2025][Model Compression][Convolutional Neural Networks] This paper proposes UniConvNet, which employs a three-layer Receptive Field Aggregator (RFA) composed of moderately sized convolution kernels (7×7, 9×9…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Convolutional Neural Networks"
  - "Effective Receptive Field"
  - "Asymptotically Gaussian Distribution"
  - "Lightweight Networks"
  - "Large-Kernel Convolution"
date: 2026-05-08
content_hash: 156f5808df386e75
---

# UniConvNet: Expanding Effective Receptive Field while Maintaining Asymptotically Gaussian Distribution for ConvNets of Any Scale

**Conference**: ICCV 2025
**arXiv**: [2508.09000](https://arxiv.org/abs/2508.09000)  
**Code**: [https://github.com/ai-paperwithcode/UniConvNet](https://github.com/ai-paperwithcode/UniConvNet)  
**Area**: Model Compression / Efficient Network Design
**Keywords**: Convolutional Neural Networks, Effective Receptive Field, Asymptotically Gaussian Distribution, Lightweight Networks, Large-Kernel Convolution

## TL;DR

This paper proposes UniConvNet, which employs a three-layer Receptive Field Aggregator (RFA) composed of moderately sized convolution kernels (7×7, 9×9, 11×11) to expand the Effective Receptive Field (ERF) while preserving its Asymptotically Gaussian Distribution (AGD), achieving consistent improvements over existing CNNs and ViTs across lightweight to large-scale model regimes.

## Background & Motivation

Large-kernel convolutional networks (e.g., SLaK, UniRepLKNet) can achieve larger effective receptive fields but suffer from two critical issues:

**High parameter and computational cost**: Extremely large kernels introduce significant parameter and FLOPs overhead.

**Disruption of the AGD of ERF**: Large-kernel convolutions cause the multi-scale influence distribution of the ERF to deviate from the natural intuition that pixels closer to the output should exert greater influence.

Traditional small-kernel networks (e.g., ResNet-101) naturally satisfy AGD through multi-scale gradient influence when stacking 3×3 convolutions, albeit with a limited ERF. The core problem addressed in this paper is: **Can smaller convolution kernels be combined appropriately to simultaneously enlarge the ERF and preserve AGD?**

## Method

### Overall Architecture

UniConvNet adopts a four-stage pyramid structure (stem + 4 stages), where each stage consists of stacked Three-layer RFA modules. The overall architecture is built upon InternImage, replacing its convolutions with the proposed RFA modules and incorporating DCNV3 residual connections (with softmax normalization removed).

### Key Designs

1. **Receptive Field Aggregator (RFA)**:

    - The input is split along the channel dimension into $N+1$ heads: $A_1, H_1, \ldots, H_N$.
    - $A_1$ is first fed into Layer Operator 1; the channel count of output $A_2$ grows from $\frac{C}{N+1}$ to $\frac{2C}{N+1}$.
    - $A_n$ is recursively passed into subsequent Layer Operators with pyramid-increasing channel dimensions, reducing parameters and FLOPs.
    - The remaining $H_n$ heads interact with $A_n$ at each layer via 1×1 convolution projections to enhance feature diversity.
    - **Design Motivation**: Assign discriminative influence to receptive fields of different scales directly within shallow modules.

2. **Layer Operator (LO)**:

    - **Amplifier (Amp)**: Applies a depthwise separable large-kernel $K \times K$ convolution followed by GELU activation to $a_{n,1}$, then performs element-wise multiplication with $a_{n,2}$. This expands the receptive field and amplifies the influence of salient pixels.
    - **Discriminator (Dis)**: Fuses features from depthwise separable $K \times K$ and $k \times k$ ($k=3$) convolutions, introducing discriminative influence from small-scale new pixels into the large receptive field.
    - The two branches are concatenated to form outputs with two-layer AGD, with progressively increasing channel counts.
    - **Design Motivation**: Construct a spatial encoder from a receptive field perspective, amplifying salient features via multiplication while incorporating local detail.

3. **Three-layer RFA Configuration**:

    - For 224×224 inputs, $N=3$ layers are used with kernel sizes $K = 2n+5$, i.e., 7×7, 9×9, and 11×11.
    - The small kernel size is $k=3$, ultimately forming a four-layer AGD receptive field.
    - The maximum kernel size of 11×11 ensures that corner pixels on the 14×14 feature maps in stage 3 have at most one-quarter overlap.
    - Stacking multiple RFA modules continuously expands the ERF while maintaining AGD.

### Loss & Training

- ImageNet-1K training for 300 epochs using the AdamW optimizer.
- Large models (UniConvNet-L/XL) are first pre-trained on ImageNet-22K for 90 epochs, then fine-tuned on ImageNet-1K for 20 epochs.
- Standard training protocols are adopted for downstream tasks (COCO detection, ADE20K segmentation).

## Key Experimental Results

### Main Results — ImageNet-1K Classification

| Model | Params | FLOPs | Top-1 Acc |
|-------|--------|-------|-----------|
| UniRepLKNet-A | 4.4M | 0.6G | 77.0% |
| **UniConvNet-A** | **3.4M** | **0.589G** | **77.0%** |
| DCNV4 | 5.3M | 0.805G | 78.5% |
| **UniConvNet-P0** | **5.2M** | **0.832G** | **79.1%** |
| ConvNeXt-T | 29.0M | 5.0G | 82.1% |
| InternImage-T | 30.0M | 5.0G | 83.5% |
| **UniConvNet-T** | **30.3M** | **5.1G** | **84.2%** |
| InternImage-B | 97.0M | 16.0G | 84.9% |
| **UniConvNet-B** | **97.6M** | **15.9G** | **85.0%** |
| InternImage-XL† | 335M | 163G | 88.0% |
| **UniConvNet-XL†** | **226.7M** | **115.2G** | **88.4%** |

### Ablation Study — Kernel Size Selection

| Model | Kernel Sizes | Params | FLOPs | Acc |
|-------|-------------|--------|-------|-----|
| UniConvNet-A | 5,7,9 | 3.5M | 0.564G | 76.6% |
| **UniConvNet-A** | **7,9,11** | **3.4M** | **0.589G** | **77.0%** |
| UniConvNet-A | 9,11,13 | 3.5M | 0.579G | 76.9% |
| UniConvNet-T | 5,7,9 | 30.0M | 5.0G | 84.1% |
| **UniConvNet-T** | **7,9,11** | **30.3M** | **5.1G** | **84.2%** |

### Key Findings

- UniConvNet-T achieves 84.2% top-1 accuracy with 30M parameters and 5.1G FLOPs, outperforming models of comparable scale by at least 0.6 percentage points.
- UniConvNet-XL surpasses CNN performance bottlenecks, reaching 88.4% top-1 accuracy with significantly fewer parameters and FLOPs than competing models at the same scale.
- Strong downstream performance: 55.7 AP$^b$ on COCO detection (Cascade Mask R-CNN) and 55.1 mIoU on ADE20K segmentation (UperNet).
- The 7,9,11 kernel size combination is optimal for 224×224 inputs; performance degrades with both larger and smaller configurations.
- Consistent improvements are observed across lightweight to large-scale variants, validating the generality of the approach.

## Highlights & Insights

1. **Theoretically Grounded**: This work is the first to interpret the strengths and weaknesses of small- and large-kernel networks through the lens of AGD in ERF, proposing a new paradigm of "expanding ERF while preserving AGD."
2. **Principled Design**: The pyramid channel-increasing strategy and recursive structure enable small kernels to achieve effects comparable to large kernels, with substantially reduced parameters and computation.
3. **Strong Generality**: Model variants ranging from 3.4M to 226.7M parameters all perform competitively, covering the full spectrum from mobile to server-side deployment.
4. **Plug-and-Play**: The Three-layer RFA can serve as a drop-in replacement for convolutions in any ConvNet.

## Limitations & Future Work

- The three-layer RFA configuration with $N=3$ is empirically determined for 224×224 inputs; the optimal configuration for higher-resolution inputs remains underexplored.
- The design is built upon the InternImage backbone, and its effectiveness under alternative backbone architectures is uncertain.
- Comparisons with recent state space models such as Mamba are absent.
- While the AGD property of ERF is demonstrated through visualization, a rigorous mathematical proof is lacking.

## Related Work & Insights

- The method contrasts with large-kernel approaches such as UniRepLKNet, demonstrating that "intelligently combining small kernels" is more effective than "simply scaling up kernel size."
- The Amplifier + Discriminator design philosophy may inspire other tasks requiring multi-scale feature fusion.
- The pyramid channel-increasing strategy is applicable to other module designs aiming to reduce computational cost.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Re-examines ConvNet design from the AGD perspective of ERF and proposes a novel design paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers classification, detection, and segmentation; includes multiple model variants from lightweight to large-scale with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and figures are intuitive, with minor typographical issues in some equations.
- **Value**: ⭐⭐⭐⭐ Offers a new perspective on CNN design and a strong baseline, with practical utility for both mobile and server-side applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Gradient Short-Circuit: Efficient Out-of-Distribution Detection via Feature Intervention](gradient_short-circuit_efficient_out-of-distribution_detection_via_feature_inter.md)
- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](../../NeurIPS2025/model_compression/4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](../../NeurIPS2025/model_compression/towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)

</div>

<!-- RELATED:END -->
