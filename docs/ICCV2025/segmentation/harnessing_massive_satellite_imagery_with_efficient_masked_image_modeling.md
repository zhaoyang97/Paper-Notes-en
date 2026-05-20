---
title: >-
  [Paper Note] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling
description: >-
  [ICCV 2025][Segmentation][Remote Sensing Foundation Model] This paper proposes a remote sensing model pre-training pipeline comprising OpticalRS-13M, a dataset of 13 million optical remote sensing images…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Remote Sensing Foundation Model"
  - "Masked Image Modeling"
  - "Large-Scale Dataset"
  - "Efficient Pre-training"
  - "SelectiveMAE"
date: 2026-05-08
content_hash: 6b1c602d4841a374
---

# Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling

**Conference**: ICCV 2025
**arXiv**: [2406.11933](https://arxiv.org/abs/2406.11933)  
**Code**: [SelectiveMAE](https://github.com/fengxiangwang/SelectiveMAE)  
**Area**: Image Segmentation
**Keywords**: Remote Sensing Foundation Model, Masked Image Modeling, Large-Scale Dataset, Efficient Pre-training, SelectiveMAE

## TL;DR

This paper proposes a remote sensing model pre-training pipeline comprising OpticalRS-13M, a dataset of 13 million optical remote sensing images, and SelectiveMAE, an efficient MIM method that selectively encodes and reconstructs patches based on semantic richness. Using only 40% of image patches, SelectiveMAE achieves performance comparable to full-patch training while delivering more than 2× speedup.

## Background & Motivation

The development of remote sensing foundation models (RSFMs) relies on large-scale self-supervised pre-training, with masked image modeling (MIM) as the core paradigm. However, two major bottlenecks exist in the remote sensing domain:

**Insufficient dataset scale and diversity**: Existing remote sensing datasets (e.g., MillionAID with ~1M images) are far smaller than natural image datasets (e.g., ImageNet-21k with ~14M images). Moreover, they primarily focus on scene-level classification, lacking fine-grained annotations for object detection and pixel-level segmentation, which limits the ability of MIM to learn generalizable representations.

**Low MIM training efficiency**: Conventional MAE reconstructs all masked patches (typically 75%), yet a distinctive characteristic of remote sensing imagery is sparse foreground and redundant background. Encoding and reconstructing large numbers of semantically uninformative background patches introduces unnecessary computational overhead. For instance, pre-training ViT-B on 1M remote sensing images requires 107 hours on 8×A100 GPUs; scaling to tens of millions of images is prohibitively expensive.

**Key Challenge**: How can one simultaneously scale data to improve representation quality and reduce the computational cost of MIM in remote sensing scenarios?

**Key Insight**: The authors address two questions: (1) Is it necessary to reconstruct all redundant background patches? (2) Can the visible patch ratio in the encoder be further reduced (e.g., from ≤25% to ≤15%)? Based on these questions, they propose a selective encoding and reconstruction strategy.

## Method

### Overall Architecture

The proposed pipeline consists of two core components:
- **OpticalRS-13M Dataset Construction**: Collection, filtering, cropping, and deduplication to form 13 million optical remote sensing images.
- **SelectiveMAE Efficient Pre-training**: HOG features are used to quantify patch semantic richness, enabling selective encoding and partial reconstruction.

### Key Designs

#### 1. OpticalRS-13M Dataset

Publicly available remote sensing datasets from nearly a decade are collected following the DiRS principles (Diversity, Richness, Scalability), with the following preprocessing steps:
- **Filtering**: Only visible-light images are retained; multispectral and SAR data are excluded.
- **Cropping**: High-resolution images are randomly cropped into sub-images ranging from 64×64 to 1024×1024 pixels.
- **Deduplication**: A two-stage deduplication pipeline — coarse filtering via perceptual hashing followed by manual inspection.

The resulting dataset covers 12 major categories (including an "events" category such as fire and flooding), is at least 4× larger than prior datasets, and exhibits richer feature distributions as visualized by t-SNE.

#### 2. Partial Reconstruction

Standard MAE applies a 75% masking ratio and reconstructs all masked patches. SelectiveMAE introduces a reconstruction ratio $r$ (default 25%), reconstructing only the semantically richest patches:

- HOG features are computed for each patch to quantify its gradient orientation histogram values.
- Patches are ranked by HOG score, and the top $\lfloor r \times N \rfloor$ patches are selected for reconstruction.
- The decoder adopts the lightweight cross-attention design from CrossMAE.

**Design Motivation**: Reconstructing large numbers of background patches in remote sensing images contributes minimally to representation learning. Selectively reconstructing semantically rich patches substantially improves throughput without sacrificing performance.

#### 3. Progressive Semantic Token Selection (PSTS)

Directly increasing the masking ratio to 85% (only 15% visible patches) causes gradient explosion and training instability. Inspired by curriculum learning, the PSTS module dynamically selects patches for encoding in a staged manner:

- **Initialization**: HOG is used to select a seed set $S^I$ comprising $s = (1-m)/2$ fraction of high-semantic patches.
- **Stage 1 (Near)**: Patches with minimum cosine distance to the seed set are selected — semantically similar and easier to learn.
- **Stage 2 (Far)**: Patches with maximum distance to the seed set are selected — semantically complementary and more challenging.
- **Stage 3 (Random)**: Patches are randomly selected to enhance robustness.

This "easy-to-hard" curriculum effectively prevents training collapse under high masking ratios.

### Loss & Training

- Loss function: MSE (consistent with MAE), computed only over the selected reconstruction patches.
- Learning rate is scaled by $m/r$ to match the loss variance of MAE.
- 12-layer decoder; masking ratio 85%; reconstruction ratio 25%.
- 60-epoch warmup within 800-epoch pre-training.

## Key Experimental Results

### Main Results

| Model | Backbone | Throughput/min | AID 20%/50% | RESISC-45 10%/20% | DIOR mAP50 | LoveDA mIoU |
|-------|----------|---------------|-------------|-------------------|------------|-------------|
| MAE† | ViT-B | 264k | 96.58/98.02 | 92.44/94.43 | 75.40 | 52.80 |
| SelectiveMAE† | ViT-B | **556k** | 96.90/98.12 | 93.35/94.58 | 75.70 | 53.05 |
| SelectiveMAE | ViT-L | 533k | **97.49/98.52** | **94.73/96.36** | **78.70** | 53.92 |
| OREOLE | ViT-G (914M) | - | 96.71/- | -/- | 77.40 | 54.00 |

†: Pre-trained on a 4M subset for 800 epochs. SelectiveMAE achieves state-of-the-art performance across mainstream remote sensing tasks with 2.1× the throughput of MAE.

### Ablation Study

| Method | Throughput/min | AID 20%/50% | RESISC-45 10%/20% |
|--------|---------------|-------------|-------------------|
| Adamae (2.36M) | 498k | 88.78/91.25 | 85.72/87.44 |
| Swin-B (88M) | 356k | 93.21/96.48 | 89.94/93.72 |
| HOG (parameter-free) | **556k** | 93.17/96.12 | 89.21/92.31 |

As a parameter-free method, HOG substantially outperforms learning-based alternatives (Swin-B) in speed while achieving comparable performance — 56% faster.

### Key Findings

- **40% of patches suffice**: Encoding 15% and reconstructing 25% of patches yields models on par with or superior to full MAE training.
- **Dataset diversity > quantity**: Under equal throughput, training on 3M images × 267 epochs outperforms 13M images × 67 epochs, indicating that OpticalRS-13M has high diversity and requires longer training to be fully exploited.
- **Efficiency gains increase with model size**: The speedup ratio for ViT-L (2.3×) exceeds that of ViT-B (2.1×), with GPU memory savings of 1.6–1.8×.
- **Progressive strategy is critical**: The far-near-random ordering causes gradient explosion, while near-far-random improves performance by approximately 4%.

## Highlights & Insights

1. **Problem-driven method design**: The selective encoding and reconstruction strategy naturally emerges from the intrinsic characteristic of remote sensing imagery — sparse foreground and redundant background.
2. **Parameter-free HOG**: Avoiding additional trainable modules keeps the approach simple and efficient.
3. **Elegant application of curriculum learning**: PSTS resolves training instability under high masking ratios.
4. **End-to-end pipeline**: The complete solution from dataset creation to efficient pre-training offers strong practical engineering value.

## Limitations & Future Work

- Currently limited to visible-light imagery; multispectral and SAR modalities are not covered.
- HOG, as a handcrafted feature, may fail to capture high-level semantic information.
- The stage boundaries in PSTS (near → far → random) are coarse-grained; smoother transitions could be explored.
- The transferability of SelectiveMAE to natural image domains has not been investigated.

## Related Work & Insights

- The partial reconstruction idea from CrossMAE serves as a direct inspiration; however, the authors find that randomly selecting reconstruction patches degrades performance in remote sensing, necessitating selection based on semantic richness.
- Curriculum learning provides a methodological foundation for addressing training instability under high masking ratios.
- Comprehensive comparisons against remote sensing foundation models (e.g., RVSA, OREOLE) demonstrate the competitiveness of the proposed pipeline.

## Rating

- **Novelty**: ⭐⭐⭐ — Selective reconstruction and progressive token selection are innovative, though the overall framework extends CrossMAE.
- **Technical Depth**: ⭐⭐⭐⭐ — Analysis is clear and systematic; experiments are thorough.
- **Value**: ⭐⭐⭐⭐⭐ — 2× speedup combined with a large-scale dataset offers significant practical value to the remote sensing community.
- **Writing Quality**: ⭐⭐⭐⭐ — Logic is clear; figures and tables are well-crafted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)
- [\[ICCV 2025\] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)
- [\[NeurIPS 2025\] Seg-VAR: Image Segmentation with Visual Autoregressive Modeling](../../NeurIPS2025/segmentation/seg-var_image_segmentation_with_visual_autoregressive_modeling.md)
- [\[ICCV 2025\] Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive Segmentation](inter2former_dynamic_hybrid_attention_for_efficient_high-precision_interactive_s.md)

</div>

<!-- RELATED:END -->
