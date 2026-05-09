---
title: >-
  [Paper Note] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention
description: >-
  [AAAI 2026][Segmentation][lightweight backbone] To address spatial redundancy (large homogeneous backgrounds) and channel redundancy (inefficient single feature space under extreme scale variation) in remote sensing imagery, LWGANet is proposed as a lightweight backbone that achieves efficient multi-scale feature representation via Top-K sparse global feature interaction (TGFI) and heterogeneous grouped attention (LWGA) modules, achieving SOTA across 4 remote sensing task categories on 12 datasets.
tags:
  - AAAI 2026
  - Segmentation
  - lightweight backbone
  - remote sensing
  - grouped attention
  - spatial redundancy
  - channel redundancy
date: 2026-05-08
content_hash: 122d4f2ba936b924
---

# LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention

**Conference**: AAAI 2026
**arXiv**: [2501.10040](https://arxiv.org/abs/2501.10040)
**Code**: [GitHub](https://github.com/AeroVILab-AHU/LWGANet)
**Area**: Image Segmentation
**Keywords**: lightweight backbone, remote sensing, grouped attention, spatial redundancy, channel redundancy

## TL;DR
To address spatial redundancy (large homogeneous backgrounds) and channel redundancy (inefficient single feature space under extreme scale variation) in remote sensing imagery, LWGANet is proposed as a lightweight backbone that achieves efficient multi-scale feature representation via Top-K sparse global feature interaction (TGFI) and heterogeneous grouped attention (LWGA) modules, achieving SOTA across 4 remote sensing task categories on 12 datasets.

## Background & Motivation

### Limitations of Prior Work

**Background**: Deep learning analysis of remote sensing images faces two inherent redundancies: (1) **Spatial redundancy** — sparse foreground objects distributed over large homogeneous backgrounds (roads, farmland, ocean), causing dense computation to waste resources on low-semantic-value background regions; (2) **Channel redundancy** — extreme scale variation in remote sensing imagery makes it difficult for a single unified feature representation to efficiently capture both fine-grained textures and macro spatial context simultaneously, resulting in many channels being irrelevant to objects at the current scale.

Existing lightweight backbones (e.g., MobileNetV2, EfficientFormerV2), designed primarily for natural images, adopt homogeneous grouping strategies (e.g., depthwise separable convolution) that apply identical operations across all channel groups and fail to effectively handle channel redundancy in remote sensing data. Convolutional models (FasterNet) offer strong local representations but lack global context, while Transformer models (EfficientFormerV2) excel at global modeling but suppress high-frequency spatial information.

### Paper Goals

**Goal**: How to design a lightweight backbone that simultaneously addresses spatial and channel redundancy in remote sensing imagery, achieving an optimal accuracy-efficiency trade-off?

## Method

### Overall Architecture
LWGANet adopts a four-stage hierarchical architecture with spatial resolutions downsampled by factors of 4/8/16/32. Three variants are provided — L0/L1/L2 (stem channel sizes 32/64/96) — with block counts $[N_1, N_2, N_3, N_4] = [1,2,4,2]$ (L0/L1) or $[1,4,4,2]$ (L2). Each LWGA Block consists of an LWGA module + Channel MLP + residual connection.

### Key Designs

**LWGA Module (Heterogeneous Grouped Attention)**
The channels are evenly split into 4 non-overlapping paths $\{\mathbf{X_1}, \mathbf{X_2}, \mathbf{X_3}, \mathbf{X_4}\}$, each with $C/4$ channels, routed to dedicated operators optimized for different scales:
- **GPA (Gate Point Attention)**: Expands to $C$ channels via $1{\times}1$ convolution then restores, generating point-wise attention via sigmoid to capture fine-grained details.
- **RLA (Regular Local Attention)**: $3{\times}3$ convolution + BN + activation, leveraging convolutional inductive bias to capture local textures.
- **SMA (Sparse Medium-range Attention)**: TGFI downsamples to $H/3 \times W/3$, cross-shaped attention (window $L=11$) models medium-range dependencies, then interpolates back to original resolution.
- **SGA (Sparse Global Attention)**: In Stages 1–2, $5{\times}5$ grouped convolution + $7{\times}7$ dilated convolution approximates global attention; in Stage 3, TGFI + standard 4-head self-attention; in Stage 4, dense self-attention is applied directly on the full feature map.

The four-path outputs are concatenated and fused into $\mathbf{Y} \in \mathbb{R}^{H \times W \times C}$.

**TGFI Module (Top-K Global Feature Interaction)**
Addressing spatial redundancy: (1) The feature map is partitioned into non-overlapping regions, and the most salient token (Top-K) is selected from each region while its spatial coordinates $\mathcal{P}_{loc}$ are recorded; (2) interactions are performed only on the sparsely sampled token set; (3) the enhanced representations are restored to their original positions via $\mathcal{P}_{loc}$.

## Key Experimental Results

**Scene Classification (NWPU/AID/UCM)**:

### Main Results

| Method | Params(M) | FLOPs(G) | NWPU Top-1 | AID | UCM |
|--------|-----------|----------|------------|-----|-----|
| MobileNet V2 1.0x | 2.28 | 0.319 | 95.06 | 93.65 | 97.14 |
| EfficientFormerV2 S0 | 3.36 | 0.396 | 94.52 | 93.80 | 97.14 |
| **LWGANet-L0** | **1.72** | **0.186** | **95.49** | **94.60** | **98.57** |
| MobileViT S | 5.03 | 1.75 | 95.19 | 95.25 | 97.14 |
| **LWGANet-L2** | 13.0 | 1.87 | **96.17** | **95.45** | **98.57** |

**Oriented Object Detection (DOTA 1.0/1.5/DIOR-R)**: LWGANet-L2 achieves a mean mAP of 73.49 across three datasets with 29.2M parameters, surpassing PKINet-S (72.30) and DecoupleNet-D2 (72.09).

**Semantic Segmentation**: mIoU of 69.1 on UAVid (+1.3 vs. ResNet18) and 53.6 on LoveDA (+1.2 vs. UnetFormer).

**Change Detection**: A2Net-LWGANet-L0 achieves a mean IoU of 83.49 across LEVIR/WHU/CDD/SYSU with only 2.91M parameters, outperforming A2Net (82.86).

**Inference Speed**: LWGANet-L0 achieves 13,234 FPS on GPU and 80 FPS on CPU, far exceeding EfficientFormerV2 (1,299 GPU FPS).

## Highlights & Insights
- Explicitly identifies and simultaneously addresses both spatial and channel redundancy as core bottlenecks in remote sensing imagery.
- Heterogeneous grouping strategy: different channel groups use operators tailored to different scales, yielding higher efficiency than homogeneous grouping.
- The sparse sampling mechanism of TGFI reduces global modeling complexity from dense to sparse.
- The progressive strategy of SGA (convolutional approximation → sparse attention → dense attention) elegantly adapts to different stages.
- Comprehensive validation across 12 datasets and 4 task categories, covering classification, detection, segmentation, and change detection.

## Limitations & Future Work
- The Top-K selection strategy is based on maximum activation values, potentially missing low-contrast but semantically important foreground regions.
- The fixed four-way equal channel grouping leaves room for exploration of adaptive or NAS-searched grouping ratios.
- Only ImageNet-1K pretraining is explored; the effect of remote sensing-specific pretraining remains uninvestigated.
- Performance on very high resolution imagery (e.g., 4K remote sensing images) under real deployment conditions has not been evaluated.

## Related Work & Insights
Compared to homogeneous lightweight models such as MobileNetV2, LWGANet significantly improves multi-scale representation capacity through heterogeneous grouping. Relative to FasterNet (pure CNN), it achieves +2.19% on NWPU at comparable speed. Compared to EfficientFormerV2 (hybrid architecture), it leads substantially in both accuracy and speed. The key distinction of this work lies in being the first to design a remote sensing-specific lightweight backbone from the perspective of **data redundancy**.

## Related Work & Insights
- The heterogeneous grouped attention paradigm is generalizable to other tasks requiring multi-scale features (e.g., medical imaging, industrial inspection).
- The sparse interaction idea of TGFI is conceptually related to efficiency optimization directions such as sparse attention and token pruning.
- The progressive global modeling strategy (from approximation to exact computation) can be adopted for other hierarchical architecture designs.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](../../CVPR2026/segmentation/sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[AAAI 2026\] SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation](ssr_semantic_and_spatial_rectification_for_clip-based_weakly_supervised_segmenta.md)

<!-- RELATED:END -->
