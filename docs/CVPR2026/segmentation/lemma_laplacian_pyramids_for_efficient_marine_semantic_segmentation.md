---
title: >-
  [Paper Note] LEMMA: Laplacian Pyramids for Efficient Marine Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Lightweight semantic segmentation] This paper proposes LEMMA, a lightweight marine semantic segmentation model based on Laplacian pyramids, which replaces deep feature computation with pyramid-decomposed edge information. LEMMA achieves SOTA-level segmentation accuracy (98.97% mIoU on MaSTr1325) with a 71× reduction in parameter count.
tags:
  - CVPR 2026
  - Segmentation
  - Lightweight semantic segmentation
  - Laplacian pyramid
  - marine semantic segmentation
  - edge detection
  - unmanned surface vehicle
date: 2026-05-08
content_hash: 49b8805dab701395
---

# LEMMA: Laplacian Pyramids for Efficient Marine Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.25689](https://arxiv.org/abs/2603.25689)
**Code**: Unavailable
**Area**: Semantic Segmentation
**Keywords**: Lightweight semantic segmentation, Laplacian pyramid, marine semantic segmentation, edge detection, unmanned surface vehicle

## TL;DR

This paper proposes LEMMA, a lightweight marine semantic segmentation model based on Laplacian pyramids, which replaces deep feature computation with pyramid-decomposed edge information. LEMMA achieves SOTA-level segmentation accuracy (98.97% mIoU on MaSTr1325) with a 71× reduction in parameter count.

## Background & Motivation

Semantic segmentation of marine scenes is critical for autonomous navigation of unmanned surface vehicles (USVs) and coastal earth observation tasks such as oil spill detection. However, existing segmentation methods (e.g., WaSR-T, DeepLabv3) typically rely on deep CNN or Transformer architectures with tens to hundreds of millions of parameters and prohibitive computational costs, making real-time deployment on resource-constrained edge devices such as UAVs and USVs infeasible.

- **Key Challenge**: Marine scenes demand high-precision segmentation (e.g., low-contrast regions such as water surface reflections and thin oil films), yet deployment platforms (UAVs/USVs) offer extremely limited computational resources. Existing methods fail to reconcile accuracy and efficiency—WaSR-T achieves 99.80% mIoU but requires 71.4M parameters and 133.8 GFLOPs.

- **Key Insight**: The paper exploits the edge information naturally provided by Laplacian pyramid decomposition. Each pyramid level encodes edge details at a specific resolution, which can be injected at early stages of feature extraction, thereby avoiding expensive feature map computation in deep layers. **Core Idea**: Replace deep feature extraction with edge priors from the Laplacian pyramid to simultaneously achieve lightweight design and high accuracy.

## Method

### Overall Architecture

LEMMA decomposes the input image into a depth-3 Laplacian pyramid ($L_1$, $L_2$, $L_3$) and processes multi-scale features through three branches: the Low-level Feature Branch (LFB) processes $L_3$ at the lowest resolution; the Middle-level Feature Branch (MFB) fuses $L_2$ with LFB outputs; and the High-level Feature Branch (HFB) integrates $L_1$ with features from the preceding branches to produce the final segmentation mask. Within each branch, residual block chains and convolutional layers perform feature extraction, while concatenation and transposed convolutions enable cross-scale information fusion.

### Key Designs

1. **Laplacian Pyramid Decomposition**:
   - **Function**: Decomposes the image into multi-resolution edge representations.
   - **Mechanism**: Each pyramid level naturally encodes high-frequency edge details at the corresponding resolution, yielding multi-scale edge representations in a single decomposition step.
   - **Design Motivation**: Edge information is a critical cue for distinguishing water surfaces, obstacles, and oil spills in marine scenes; the pyramid avoids the high cost of progressively learning edge features in deep networks.

2. **Three-Branch Residual Architecture (LFB/MFB/HFB)**:
   - **Function**: Refines and fuses pyramid-level features at varying depths.
   - **Mechanism**: LFB processes the lowest-resolution features (64 channels); MFB fuses low- and mid-level information; HFB reconstructs the mask at the highest resolution using only 16 channels.
   - **Design Motivation**: Using 16 instead of 64 channels in HFB substantially reduces GFLOPs on high-resolution feature maps; concatenation preserves original information from each level, avoiding information loss.

3. **Configurable Residual Block Chains**:
   - **Function**: Controls the feature extraction depth within each branch.
   - **Mechanism**: Each branch embeds a configurable number of residual blocks (NRBL/NRBM/NRBH), each consisting of conv–LeakyReLU–conv with a residual connection.
   - **Design Motivation**: Ablation studies identify the optimal configuration per dataset (MaSTr1325: 7/7/1; Oil Spill: 6/7/4), achieving the best parameter–accuracy trade-off.

### Loss & Training

- Focal Loss is adopted as the training objective and outperforms Dice Loss and CE+Dice combinations on both datasets.
- Adam optimizer is used with a batch size of 8 for 300 epochs.
- Training is conducted on an NVIDIA TESLA P100; inference is evaluated on an NVIDIA 2080 GPU and an Intel 4-core XEON CPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (LEMMA) | Prev. SOTA | Gain |
|---------|--------|--------------|------------|------|
| MaSTr1325 | mIoU | 98.97% | 99.91% (BEMRF-Net) | −0.94% (but 71× fewer params) |
| MaSTr1325 | Parameters | 1.07M | 71.4M (WaSR-T) | 66.7× reduction |
| MaSTr1325 | GFLOPs | 17.83 | 156.0 (BEMRF-Net) | 88.5% reduction |
| MaSTr1325 | Inference time | 7.3 ms | 47.55 ms (DeepLabv3) | 84.65% reduction |
| Oil Spill | mIoU | 93.42% | 92.66% (R-GSSNet) | +0.76% |
| Oil Spill | Parameters | 1.01M | 62.6M (R-Segformer) | 62× reduction |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Residual blocks 7/7/1 (MaSTr1325) | mIoU 98.96% | Optimal; adding more HFB blocks degrades performance |
| Residual blocks 6/7/4 (Oil Spill) | mIoU 93.42% | Optimal configuration |
| Focal Loss vs. Dice Loss | 98.97% vs. 98.72% | Focal Loss is superior on both datasets |
| Focal Loss vs. CE+Dice | 98.97% vs. 98.86% | Confirms the advantage of Focal Loss |

### Key Findings

- With approximately 1M parameters, LEMMA achieves performance competitive with models possessing tens of millions of parameters (e.g., WaSR-T at 71.4M).
- The model performs well across two substantially different viewpoints—USV ground-level perspective (MaSTr1325) and UAV aerial perspective (Oil Spill)—demonstrating cross-platform robustness.
- Using only 16 channels in HFB is sufficient for high-resolution mask reconstruction, which is the key design choice for reducing computational cost.
- The Laplacian pyramid implicitly suppresses low-frequency illumination artifacts such as sun glare and water surface reflections.

## Highlights & Insights

- The paper effectively combines a classical image processing technique (Laplacian pyramid) with deep residual networks, leveraging physical priors to reduce the learning burden.
- Extreme lightweight design: approximately 1M parameters suffice to achieve near-SOTA accuracy, enabling real-time deployment on resource-constrained edge devices such as UAVs and USVs.
- Strong cross-platform generality: the same architecture applies to both ground-level USV obstacle detection and aerial oil spill segmentation.
- No ImageNet pretraining is required; high performance is achieved by training from scratch.

## Limitations & Future Work

- Environmental factors such as reflections, waves, and glare degrade the quality of the Laplacian pyramid and cause failures (failure cases due to reflections are shown in the paper).
- The current design uses a fixed pyramid depth and static residual block configurations; adaptive pyramid depth allocation warrants future investigation.
- Dataset scales are limited (MaSTr1325: 1,325 images; Oil Spill: 847 images), making it difficult to verify generalization to large-scale scenarios.
- An accuracy gap of approximately 1% remains relative to the strongest models such as WaSR-T.

## Related Work & Insights

- **vs. WaSR-T**: WaSR-T achieves 99.80% mIoU with Transformer-based design but requires 71.4M parameters; LEMMA achieves 98.97% with 1.07M parameters, offering orders-of-magnitude improvement in efficiency.
- **vs. DeepLabv3**: DeepLabv3 achieves 97.67% mIoU with 48M parameters and 123 GFLOPs; LEMMA surpasses it with 1/45 of the parameters.
- **vs. LETNet**: Another lightweight model, LETNet achieves 83.18% mIoU; LEMMA improves performance by nearly 16 percentage points with a comparable parameter count (1.07M vs. 0.94M).
- **Insight**: Integrating classical CV techniques (pyramids, edge detection) with deep learning can achieve extreme lightweight design in domain-specific applications.

## Rating

- **Novelty**: ⭐⭐⭐ — Laplacian pyramids for segmentation are not entirely new, but the three-branch design and its application to marine scenes are noteworthy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, extensive baseline comparisons, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated design, and thorough experimental analysis.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to marine segmentation on edge devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](../../ACL2026/segmentation/boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)

</div>

<!-- RELATED:END -->
