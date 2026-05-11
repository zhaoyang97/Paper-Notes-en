---
title: >-
  [Paper Note] Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter
description: >-
  [AAAI2026][Segmentation][underwater instance segmentation] This paper is the first to introduce DINOv2 into underwater instance segmentation. Through two adaptation modules—AquaStyle Aligner (Fourier frequency-domain sty…
tags:
  - "AAAI2026"
  - "Segmentation"
  - "underwater instance segmentation"
  - "DINOv2"
  - "domain adaptation"
  - "Fourier style transfer"
  - "foundation model fine-tuning"
date: 2026-05-08
content_hash: 60e076d109f5cc9e
---

# Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter

**Conference**: AAAI2026
**arXiv**: [2511.08334](https://arxiv.org/abs/2511.08334)
**Code**: [ettof/Diveseg](https://github.com/ettof/Diveseg)
**Area**: Image Segmentation
**Keywords**: underwater instance segmentation, DINOv2, domain adaptation, Fourier style transfer, foundation model fine-tuning

## TL;DR

This paper is the first to introduce DINOv2 into underwater instance segmentation. Through two adaptation modules—AquaStyle Aligner (Fourier frequency-domain style injection) and ObjectPrior Prompter (binary mask prior prompting)—the proposed DiveSeg achieves efficient domain adaptation and substantially outperforms SAM-based methods on the UIIS and USIS10K benchmarks with fewer parameters.

## Background & Motivation

Underwater Instance Segmentation (UIS) requires simultaneous pixel-level classification and instance-level discrimination, and is a core technique for ocean exploration, ecological monitoring, and underwater robot navigation. Underwater imagery presents unique challenges:

- **Light absorption and scattering**: Long-wavelength light is absorbed by water, causing a blue-green color shift.
- **Forward scattering** induces blurring; **backscattering** reduces visibility.
- Degradation effects are non-uniform and depth-dependent, leading to large appearance variations across instances.

Early CNN-based methods (e.g., WaterMask) are limited by representational capacity. SAM-based methods (USIS-SAM) incorporate visual foundation models but rely on large-scale annotated underwater data with limited improvement. DINOv2 acquires task-agnostic general features through self-supervised learning and exhibits strong generalization, which is particularly advantageous in underwater scenarios with scarce annotations. However, PCA visualizations reveal that directly transferring DINOv2 to underwater tasks results in features heavily affected by background noise and potential target omissions.

## Core Problem

How to efficiently adapt DINOv2 to underwater scenarios from two complementary levels:

1. **Scene-level adaptation**: Eliminate the misalignment between underwater color shifts and the pretraining domain.
2. **Object-level adaptation**: Generalize the model to underwater targets such as corals, jellyfish, and sea turtles that are rarely present in the LVD-142M pretraining data.

## Method

### Overall Architecture: DiveSeg

DiveSeg is built upon a frozen DINOv2 ViT-L backbone combined with a Mask2Former segmentation head, augmented by two core adaptation modules. The ViT layers are evenly divided into four blocks; an AquaStyle Aligner is inserted at the first layer of each block, and an ObjectPrior Prompter is inserted after each block.

### AquaStyle Aligner

**Goal**: Eliminate underwater color domain shift at the scene level.

**Style Extraction**:

- The input image is transformed via the Fourier transform, separating the amplitude component (containing low-level statistics such as color information) from the phase component (containing content/structural information).
- The phase is fixed to its mean value, retaining only the amplitude; the inverse Fourier transform reconstructs a "style image" that removes object content while preserving underwater color characteristics.
- A multi-layer convolutional network followed by global average pooling encodes the style image into a compact style vector $p_x$.

**Style Injection**:

- Operating as a parallel branch to Multi-head Attention (MHA) in the ViT, a cross-attention mechanism is employed: ViT features serve as queries, while the style vector processed by an MLP serves as keys and values.
- The cross-attention output is added to the original MHA output: $\omega_1 = MHA(V_{in}) + CrossAttn(V_{in}, MLP(p_x))$
- A parallel bottleneck MLP is similarly added alongside the Feed-Forward layer for deeper feature fusion.
- All original MHA and FF parameters are frozen; only the injected module parameters are trained.

### ObjectPrior Prompter

**Goal**: Provide instance-agnostic foreground priors at the object level to reduce the learning difficulty of instance segmentation.

**Multi-scale Encoder**: Three convolutional layers with stride-2 downsampling extract a three-scale feature pyramid $\{f_M^1, f_M^2, f_M^3\}$ at resolutions $1/8^2$, $1/16^2$, and $1/32^2$.

**Pseudo-Mask Generation**: At each scale, $1\times1$ convolution followed by Sigmoid produces pseudo-masks $P_{mask}^k$, supervised by binary foreground masks obtained by merging all instance ground-truth annotations.

**Feature Enhancement**: The pseudo-mask is element-wise multiplied with the original features to filter foreground regions, then fused via convolution and residual connection: $f_{MT}^k = Conv(P_{mask}^k \cdot f_M^k) + f_M^k$

**Prior Injection**: Multi-scale enhanced features are flattened and concatenated into $O_{prompt}$, which interacts with ViT features via cross-attention ($O_{prompt}$ as keys/values, ViT features as queries); the output is added to the original ViT features before being passed to the decoder.

### Loss & Training

- Backbone: DINOv2 ViT-L (frozen)
- Decoder head: Mask2Former
- Optimizer: AdamW, weight decay 0.05, initial lr 1e-4 with warmup
- 30,000 iterations with lr decayed by $1/10$ at iterations 23,000 and 27,000
- Losses: classification loss + mask loss (Mask2Former) + BCE + IoU + L1 loss (pseudo-mask)
- Hardware: NVIDIA A100, batch size 8

## Key Experimental Results

### UIIS Dataset (7 categories, 3,937 train / 691 test)

| Method | Backbone | Params | mAP | AP50 | AP75 |
|--------|----------|--------|-----|------|------|
| WaterMask | ResNet-101 | 67M | 27.2 | 43.7 | 29.3 |
| USIS-SAM | ViT-H | 701M | 29.4 | 45.0 | 32.3 |
| **DiveSeg** | **ViT-L** | **390M** | **35.6** | **52.0** | **38.5** |

Compared with USIS-SAM: mAP +21.1%, AP50 +15.6%, AP75 +19.2%, with only 55.6% of the parameters.

### USIS10K Dataset (class-agnostic / multi-class)

| Method | Class-agnostic mAP | Multi-class mAP |
|--------|--------------------|-----------------|
| USIS-SAM (ViT-H, 701M) | 59.7 | 43.1 |
| **DiveSeg (ViT-L, 390M)** | **64.1** | **48.4** |

### Ablation Study

| Configuration | mAP | AP50 | AP75 |
|---------------|-----|------|------|
| DINOv2 + Mask2Former (baseline) | 30.9 | 44.6 | 32.2 |
| + AquaStyle Aligner | 34.1 | 50.8 | 37.8 |
| + ObjectPrior Prompter | 34.8 | 50.6 | 37.6 |
| **Full Model** | **35.6** | **52.0** | **38.5** |

### Comparison of Adaptation Strategies (alternatives to AquaStyle Aligner)

| Strategy | mAP |
|----------|-----|
| Frozen (no adaptation) | 30.9 |
| Full Fine-tuning | 31.1 |
| LoRA | 31.8 |
| Adapter | 32.7 |
| **AquaStyle Aligner** | **34.1** |

The poor performance of full fine-tuning is likely attributable to catastrophic forgetting. By explicitly modeling underwater style information, AquaStyle Aligner outperforms generic parameter-efficient fine-tuning strategies.

## Highlights & Insights

- **First introduction of DINOv2 to underwater instance segmentation**, demonstrating that self-supervised pretrained foundation models can be efficiently adapted to underwater scenarios.
- **Elegant AquaStyle Aligner design**: Fourier frequency-domain decomposition captures underwater color characteristics and injects them into the ViT via cross-attention, with clear physical intuition.
- **Insightful ObjectPrior Prompter**: Decouples instance segmentation into "foreground awareness → instance discrimination," reducing learning difficulty.
- **Exceptional parameter efficiency**: 390M parameters (ViT-L) surpass USIS-SAM with 701M parameters (ViT-H), with the majority of parameters remaining frozen.
- Qualitative results demonstrate clear advantages in challenging scenarios including fish-school segmentation under shadows, overlapping instance separation, and misclassification correction.

## Limitations & Future Work

- Validation is limited to only two datasets (UIIS and USIS10K), leaving the diversity of underwater scenarios underexplored.
- ObjectPrior Prompter relies on ground-truth merged binary masks during training but uses predicted pseudo-masks at inference; prediction quality directly affects final performance.
- Only ViT-L is evaluated; scalability to ViT-B (lighter) or ViT-G (stronger) remains unexplored.
- Inference speed and real-time capability are not discussed, leaving applicability to time-critical applications such as underwater robotics uncertain.
- Style extraction relies on globally averaged phase; style variation across different depths and water conditions may be more complex in practice.

## Related Work & Insights

| Dimension | WaterMask | USIS-SAM | DiveSeg |
|-----------|-----------|----------|---------|
| Foundation Model | CNN (ResNet) | SAM (ViT-H) | DINOv2 (ViT-L) |
| Parameters | 67M | 701M | 390M |
| Pretraining | Supervised | Supervised (mask annotations) | Self-supervised |
| Domain Adaptation | Dedicated modules | LoRA + Adapter | Style injection + prior prompting |
| Core Mechanism | End-to-end learning | Prompt engineering | Two-level adaptation (scene + object) |

The Fourier frequency-domain style transfer approach is generalizable to other domain adaptation settings (e.g., medical imaging, remote sensing), with the core insight that amplitude encodes domain-related low-level statistics. The **decoupled adaptation strategy** (scene-level + object-level) constitutes a general paradigm for transferring foundation models to domain-specific segmentation tasks. The prior injection mechanism of ObjectPrior Prompter complements SAM-style prompt engineering: the former learns implicit priors, while the latter requires explicit prompt design. In the era of foundation models, combining parameter-efficient fine-tuning with domain knowledge injection may represent the best practice for low-data domain transfer.

## Rating
- **Novelty**: 4/5 (first introduction of DINOv2 to UIS; both modules are insightfully designed)
- **Experimental Thoroughness**: 4/5 (two datasets, thorough ablations, multi-strategy comparison; speed analysis is absent)
- **Writing Quality**: 4/5 (clear structure, well-motivated)
- **Value**: 4/5 (establishes a strong baseline for underwater vision; the transfer methodology has broad applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Mixed Diet Makes DINO An Omnivorous Vision Encoder](../../CVPR2026/segmentation/a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)
- [\[CVPR 2026\] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](../../CVPR2026/segmentation/elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](../../CVPR2026/segmentation/phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](../../CVPR2026/segmentation/unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)

</div>

<!-- RELATED:END -->
