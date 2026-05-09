---
title: >-
  [Paper Note] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing
description: >-
  [AAAI 2026][Segmentation][semi-supervised learning] This paper proposes the S5 framework, which for the first time extends semi-supervised semantic segmentation into a pre-training paradigm for remote sensing foundation models (RSFMs). By constructing the million-scale RS4P-1M dataset and introducing a MoE-based multi-dataset fine-tuning strategy, S5 achieves state-of-the-art performance across multiple remote sensing segmentation and detection benchmarks.
tags:
  - AAAI 2026
  - Segmentation
  - semi-supervised learning
  - remote sensing
  - foundation model
  - semantic segmentation
  - mixture of experts
date: 2026-05-08
content_hash: c2fbee3ea4ce7ed1
---

# S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing

**Conference**: AAAI 2026
**arXiv**: [2508.12409](https://arxiv.org/abs/2508.12409)
**Code**: [S5](https://github.com/lianglyu/S5)
**Area**: Remote Sensing Image Segmentation
**Keywords**: semi-supervised learning, remote sensing, foundation model, semantic segmentation, mixture of experts

## TL;DR

This paper proposes the S5 framework, which for the first time extends semi-supervised semantic segmentation into a pre-training paradigm for remote sensing foundation models (RSFMs). By constructing the million-scale RS4P-1M dataset and introducing a MoE-based multi-dataset fine-tuning strategy, S5 achieves state-of-the-art performance across multiple remote sensing segmentation and detection benchmarks.

## Background & Motivation

Semantic segmentation in remote sensing relies heavily on pixel-level annotations, which are extremely costly to obtain. Semi-supervised semantic segmentation (S4) leverages unlabeled data through pseudo-labels and consistency regularization, yet existing S4 research remains confined to small-scale datasets and models (e.g., splitting training sets solely on iSAID), making it unable to fully exploit the vast amount of available Earth observation data.

Meanwhile, although RSFMs have made progress in self-supervised (MAE, contrastive learning) and supervised pre-training, supervised pre-training is limited by annotation scale, and self-supervised pre-training suffers from a gap with downstream segmentation tasks. The root cause is: **can S4 be scaled to millions of unlabeled remote sensing images as a pre-training strategy for RSFMs?**

The core idea of this paper is to elevate S4 from "an intra-dataset training strategy" to "a large-scale pre-training paradigm (S4P)," while ensuring pseudo-label quality through data filtering and enabling efficient multi-dataset adaptation through MoE fine-tuning.

## Method

### Overall Architecture

S5 consists of three stages: (1) Dataset Construction — filtering million-scale unlabeled images from datasets such as MillionAID via entropy filtering and diversity expansion to build RS4P-1M; (2) S4 Pre-training (S4P) — using iSAID as labeled data and RS4P-1M as unlabeled data, performing semi-supervised pre-training with FixMatch; (3) MoE Multi-Dataset Fine-tuning (MoE-MDF) — embedding FFN-MoE modules into a shared backbone, jointly fine-tuning on multiple downstream benchmarks using shared experts and dataset-specific experts.

### Key Designs

**1. Data Filtering via Entropy Filtering + Diversity Expansion**

Directly using all unlabeled data degrades pseudo-label quality due to distribution mismatch. S5 first trains an initial segmentation model (ViT-H + UperNet) on iSAID, performs inference on unlabeled images, and computes the pixel-level mean entropy:

$$E(x) = -\frac{1}{H \times W} \sum_{i=1}^{H \times W} \sum_{k=1}^{K} P^k(x^i) \log P^k(x^i)$$

Low-entropy samples are prioritized (more reliable pseudo-labels), but selecting only low-entropy samples leads to semantic redundancy. Therefore, K-Means is further applied to cluster labeled image features into $M$ clusters, unlabeled images are assigned to the nearest cluster, and selection proceeds by quota:

$$B_m^u = B^u \cdot \frac{N_m^l}{B^l}$$

Selection stops once each cluster reaches its quota, ensuring semantic diversity. The final RS4P-1M dataset contains 1 million images.

**2. S4 Pre-training Based on FixMatch**

Weak-strong augmentation consistency regularization is adopted. Two views of each unlabeled image are generated: a weakly augmented view (random scaling, cropping, flipping) and a strongly augmented view (CutMix, color jitter, Gaussian blur). The total loss is:

$$\mathcal{L} = \mathcal{L}_s + \lambda \mathcal{L}_{u_s}$$

where the supervised loss $\mathcal{L}_s$ is standard cross-entropy, and the unsupervised loss is computed only for high-confidence pseudo-labels ($\max(p_j^{u_w}) \geq \tau$):

$$\mathcal{L}_{u_s} = \frac{1}{B_u} \sum_{j=1}^{B_u} \mathbb{1}(\max(p_j^{u_w}) \geq \tau) \mathcal{L}_{ce}(\hat{y}_j, p_j^{u_s})$$

All RSFMs are initialized with MAE pre-trained weights, and S4P further enhances representation capability on top of this.

**3. MoE Multi-Dataset Fine-tuning (MoE-MDF)**

The conventional "one model per dataset" approach leads to parameter redundancy. MoE-MDF introduces a branching structure in the FFN of ViT: shared experts learn general features while dataset-specific experts learn domain-specific features. Specifically, the intermediate feature $F_{\text{FFN}}$ is passed through two parallel linear layers:

$$F^{\text{shared}} = \text{Linear}_{\text{shared}}^{D \to (1-\alpha)C}(F^{\text{FFN}})$$
$$F^{\text{specific}} = \text{Linear}_{\text{specific}}^{D \to \alpha C}(F^{\text{FFN}})$$

The outputs are concatenated as $F^{\text{out}} = \text{Concat}(F^{\text{shared}}, F^{\text{specific}})$. The hyperparameter $\alpha$ controls the ratio of shared to specific capacity, with no additional inference latency.

### Loss & Training

- Pre-training stage: Joint supervised and unsupervised FixMatch loss; low-quality pseudo-labels are filtered by confidence threshold $\tau$
- Fine-tuning stage: Dataset-specific decoders with a shared MoE backbone; shared experts are updated with all datasets, while specific experts are updated only with their corresponding dataset

## Key Experimental Results

### Main Results

| Method | Backbone | Det. Params (M) | DIOR-R mAP | DOTA-v2 mAP | Seg. Params (M) | Vaihingen mIoU | Potsdam mF1 | LoveDA mIoU | OpenEarthMap mIoU |
|------|----------|------------|-----------|------------|------------|---------------|-------------|-------------|-------------------|
| RVSA | ViT-B+RVSA | 222.4 | 68.06 | 55.22 | 412.8 | 78.49 | 91.58 | 52.44 | 66.63 |
| GFM | Swin-B | 208.2 | 67.67 | 59.15 | 387.6 | 79.61 | 91.85 | 54.98 | 67.78 |
| MTP | ViT-L+RVSA | 669.2 | 74.54 | 58.41 | 1309.6 | 80.62 | 92.47 | 54.16 | 69.04 |
| SelectiveMAE | ViT-L | 669.2 | 71.75 | 57.84 | 1309.6 | 80.45 | 92.78 | 54.31 | 69.30 |
| BillionFM | ViT-G | 1993.9 | 73.62 | 58.69 | - | - | 92.58 | 54.40 | - |
| **S5** | **ViT-B** | **138.3** | **72.95** | **57.20** | **160.4** | **79.85** | **92.40** | **54.02** | **68.65** |
| **S5** | **ViT-L** | **377.8** | **75.21** | **59.71** | **435.0** | **80.72** | **92.78** | **55.67** | **69.66** |
| **S5** | **ViT-H** | **730.0** | **75.30** | **59.89** | **824.5** | **80.85** | **92.97** | **55.65** | **70.02** |

S5-ViT-L requires only 435M parameters in the multi-dataset setting (1/3 of Scale-MAE/SelectiveMAE) while comprehensively outperforming all baselines.

### Ablation Study

**Comparison of Pre-training Datasets** (ViT-B backbone):

| Unlabeled Data | # Images | iSAID Val | Vaihingen | LoveDA | DIOR-R |
|-----------|--------|----------|-----------|--------|--------|
| None (MAE baseline) | - | 65.93 | 78.27 | 52.47 | 68.02 |
| SAMRS | 100k | 67.59 | 79.61 | 53.66 | 69.13 |
| MillionAID-random | 100k | 66.32 | 79.49 | 53.20 | 69.02 |
| MillionAID* (filtered) | 100k | **67.66** | **79.77** | **53.81** | **69.65** |

The filtered MillionAID* outperforms both random sampling and SAMRS across all tasks.

**Comparison of Fine-tuning Strategies** (MAE+S4P, ViT-B):

| Fine-tuning Strategy | Params (M) | Vaihingen | Potsdam | OpenEarthMap | LoveDA | Avg. |
|---------|---------|-----------|---------|-------------|--------|------|
| SDF (single-dataset) | 412.8 | 79.93 | 92.24 | 67.35 | 54.51 | 73.51 |
| MDF (multi-dataset) | 132.1 | 79.82 | 92.25 | 68.41 | 54.53 | 73.75 |
| MoE-MDF (α=1/4) | 160.4 | **79.85** | **92.40** | **68.80** | 54.57 | **74.15** |
| MoE-MDF (α=1/2) | 188.7 | 79.84 | 92.39 | 68.66 | **54.64** | 73.88 |

MoE-MDF achieves an average accuracy gain of 0.64 points over SDF while using only 1/3 of its parameters. α=1/4 is the optimal ratio.

### Key Findings

- S4P as a second-stage pre-training following MAE yields significant improvements across all downstream tasks
- The data filtering strategy is effective: filtered 100k images outperform random 100k and scale well to 1M
- Both model scale and data scale contribute consistently: ViT-B→ViT-H and 100k→1M both yield continuous gains
- MoE-MDF enables unified multi-dataset deployment with minimal parameter overhead

## Highlights & Insights

- **New Paradigm**: Elevating semi-supervised learning from a "training trick" to a "foundation model pre-training strategy" represents a conceptual breakthrough
- **High Practicality**: RS4P-1M dataset construction requires no additional annotations (unlike SAMRS which depends on SAM-generated masks), offering strong scalability
- **Parameter Efficiency**: MoE-MDF reduces parameter count to 1/3–1/4 of conventional methods in multi-dataset scenarios

## Limitations & Future Work

- Pre-training exclusively employs FixMatch as the S4 method; the potential gains from more recent semi-supervised methods (e.g., UniMatchV2) remain unexplored
- The MoE design is relatively simple (only shared and specific branches), without routing mechanisms or more sophisticated expert selection
- Data filtering requires an initial model trained on iSAID, introducing a dependency on the quality of this initial model

## Related Work & Insights

- **vs. MTP**: MTP relies on supervised multi-task pre-training, whereas S5 uses semi-supervised pre-training without additional annotations yet achieves superior performance (ViT-L: 75.21 vs. 74.54 mAP on DIOR-R)
- **vs. SAMRS**: SAMRS depends on SAM to generate masks for annotation construction, limiting its scale; S5 directly leverages unlabeled data, making it more scalable
- **vs. SelectiveMAE**: Both focus on data selection, but SelectiveMAE applies it to self-supervised pre-training while S5 applies it to semi-supervised pre-training

## Rating

- Novelty: ⭐⭐⭐⭐ Extending S4 into a pre-training paradigm is a novel direction; the RS4P-1M dataset construction method is practically valuable
- Experimental Thoroughness: ⭐⭐⭐⭐ Six benchmarks, multi-scale models, and detailed ablations constitute a comprehensive evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rich figures and tables, intuitive architecture diagrams
- Value: ⭐⭐⭐⭐ Provides a new pre-training paradigm for remote sensing foundation models with high practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2026/segmentation/semitooth_a_generalizable_semisupervised_framework.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [\[AAAI 2026\] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention](lwganet_addressing_spatial_and_channel_redundancy_in_remote_sensing_visual_tasks.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](../../CVPR2026/segmentation/sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[AAAI 2026\] SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation](ssr_semantic_and_spatial_rectification_for_clip-based_weakly_supervised_segmenta.md)

</div>

<!-- RELATED:END -->
