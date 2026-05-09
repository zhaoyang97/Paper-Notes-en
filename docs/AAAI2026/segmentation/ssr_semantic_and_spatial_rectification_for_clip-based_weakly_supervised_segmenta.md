---
title: >-
  [Paper Note] SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation
description: >-
  [AAAI 2026][Segmentation][Weakly supervised semantic segmentation] This paper proposes SSR, a dual-level semantic and spatial rectification framework that addresses non-target foreground over-activation caused by CLIP's cross-modal semantic misalignment via Cross-Modal Prototype Alignment (CMPA), and background over-activation during affinity propagation via Superpixel-Guided Correction (SGC). SSR achieves state-of-the-art performance on PASCAL VOC and MS COCO, surpassing both single-stage and multi-stage methods.
tags:
  - AAAI 2026
  - Segmentation
  - Weakly supervised semantic segmentation
  - CLIP
  - cross-modal prototype alignment
  - superpixel-guided correction
  - class activation map
date: 2026-05-08
content_hash: ece48df4c0e1b0b8
---

# SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation

**Conference**: AAAI 2026
**arXiv**: [2512.01701](https://arxiv.org/abs/2512.01701)
**Code**: None
**Area**: Segmentation
**Keywords**: Weakly supervised semantic segmentation, CLIP, cross-modal prototype alignment, superpixel-guided correction, class activation map

## TL;DR

This paper proposes SSR, a dual-level semantic and spatial rectification framework that addresses non-target foreground over-activation caused by CLIP's cross-modal semantic misalignment via Cross-Modal Prototype Alignment (CMPA), and background over-activation during affinity propagation via Superpixel-Guided Correction (SGC). SSR achieves state-of-the-art performance on PASCAL VOC and MS COCO, surpassing both single-stage and multi-stage methods.

## Background & Motivation

Weakly supervised semantic segmentation (WSSS) aims to generate high-quality pseudo-labels using only image-level annotations, thereby avoiding the prohibitive cost of pixel-level labeling. Existing methods typically follow a three-stage pipeline: (1) train a classification network to generate initial CAMs; (2) refine the CAMs; (3) generate pseudo-labels to train a segmentation model.

CLIP has been widely adopted in WSSS due to its strong cross-modal semantic understanding and GradCAM-based initial CAM generation, substantially outperforming conventional CNN and ViT approaches. Nevertheless, CLIP-based methods still face **two core challenges**:

**Over-activation of non-target foreground regions**: This stems from CLIP's inherent modality gap. Visual features focus on low-level patterns (color, shape), whereas text features encode high-level abstract semantics, leading to semantic misalignment. Existing methods address this only through prompt engineering, without fundamentally bridging the cross-modal representational gap.

**Over-activation of background regions**: During feature refinement, anomalously high affinity values between background and target regions produce spurious background responses. Existing approaches rely on multi-stage iterative refinement or affinity matrix constraints, yet remain susceptible to low-level feature interference and global context confusion.

The root causes of these two issues reside at the semantic level and the spatial level, respectively, motivating the authors to design a coordinated dual-dimensional modeling framework.

## Method

### Overall Architecture

The SSR framework takes image modality $I$ and text modality $T$ as inputs, where $T$ comprises $K$ foreground categories and $M$ background categories. The framework consists of two core modules:
- **Semantic level**: Cross-Modal Prototype Alignment (CMPA), which reduces the modality gap via contrastive learning between image and text prototypes.
- **Spatial level**: Superpixel-Guided Correction (SGC), which leverages superpixel spatial priors to filter noise in the affinity matrix.

### Key Designs

#### 1. **Cross-Modal Prototype Alignment (CMPA)**

**Core Idea**: Establish a contrastive learning mechanism over cross-modal positive and negative pairs to jointly optimize modality alignment and classification boundaries.

**Multi-modal prototype construction**: For $N$ image-text pairs, structurally identical but parameter-independent ISA and TSA modules are used to project visual and textual features into a unified space:

$$v_i' = \text{ISA}(v_i), \quad t_i' = \text{TSA}(t_i)$$

GradCAM is then used to generate $CAM_c$, and masked average pooling (MAP) is applied to compute foreground image and text features:

$$f_{image} = MAP(CAM^c \odot v_i'), \quad f_{text} = t_i'[index]$$

Foreground features collected across all samples are clustered via K-means to obtain image prototypes $P^I \in \mathbb{R}^{K \times d_2}$ and text prototypes $P^T \in \mathbb{R}^{K \times d_2}$.

**Prototype contrastive learning**: Fine-grained semantic alignment is achieved through three constraints:
- Visual features are attracted toward text prototypes of the same category.
- Text prototypes aggregate visual prototypes of the same category.
- Cross-modal prototypes of different categories are mutually repelled.

The contrastive loss is defined as:

$$\mathcal{L}_{proto} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(S_{pos}^i)}{\exp(S_{pos}^i) + \sum_{j=1}^{k}\exp(S_{neg_j}^i)}$$

where the temperature hyperparameter $\tau_{proto}$ is set as a learnable parameter. The **Design Motivation** is that the ISA/TSA projection space preserves CLIP's instance discrimination capability while reducing prototype construction cost through dimensionality reduction.

#### 2. **Superpixel-Guided Correction (SGC)**

**Core Idea**: Exploit superpixel structural information to construct a binary mask that selectively suppresses column vectors in the affinity matrix associated with non-target regions, thereby inhibiting erroneous propagation of background semantics.

**Superpixel clustering**: The SLIC algorithm is applied for superpixel segmentation (with substantially lower computational overhead than SAM), followed by color-space-based clustering to identify target regions:

$$C = \text{K-means}(\text{SLIC}(I_i))$$

The proportion of high-confidence pixel activations within each cluster region is computed; clusters exceeding a threshold are designated as target regions, forming binary mask matrix $Mask$.

**Affinity matrix correction**: CLIP's global semantics and DINO's local spatial relationships are fused:

$$A = ConCat(MHSA_{CLIP}, MHSA_{DINO})$$

CLIP provides high-level semantic guidance while DINO supplies fine-grained spatial relationships; the fused result is normalized. The mask is then applied to refine the affinity matrix and enhance the initial CAM:

$$A^* = A \odot Mask, \quad CAM_{refine}^c = A^* \otimes CAM^c$$

#### 3. **End-to-End Segmentation Optimization**

The overall training objective combines the prototype contrastive loss and the segmentation loss.

### Loss & Training

$$\mathcal{L}_{SSR} = \mathcal{L}_{proto} + \gamma \mathcal{L}_{seg}$$

- $\mathcal{L}_{proto}$: Cross-modal prototype contrastive loss, encouraging same-category cross-modal features to cluster together and different-category features to separate.
- $\mathcal{L}_{seg}$: Cross-entropy loss using online-generated pseudo-masks for end-to-end training.
- Loss weight $\gamma = 0.1$; prototype temperature $\tau_{Proto} = 0.05$.
- Prototypes are updated every 5,000 iterations.
- CLIP-to-DINO weight ratio: 0.4:0.6.

## Key Experimental Results

### Main Results

| Dataset | Metric | SSR | SSR (w/o CRF) | ExCEL | VPL | WeCLIP | MoRe |
|--------|------|-----|--------------|-------|-----|--------|------|
| VOC Val | mIoU | **79.5** | 78.2 | 78.4 | 79.3 | 76.4 | 76.4 |
| VOC Test | mIoU | **79.6** | 78.1 | 78.5 | 79.0 | 77.2 | 75.0 |
| COCO Val | mIoU | **50.6** | 49.2 | 50.3 | 49.8 | 47.1 | 47.4 |

As a single-stage method, SSR surpasses all single-stage methods as well as multi-stage methods such as VPL. On VOC val, it achieves 97.4% of fully supervised performance. In terms of CAM seed quality, SSR attains 78.7% mIoU on VOC train, exceeding the previous state of the art by at least 0.7%.

### Ablation Study

| Configuration | P | R | mIoU | Note |
|------|---|---|------|------|
| CMPA only | 72.8 | 84.6 | 63.3 | Initial CAM |
| + CLIP attention | 85.2 | 88.9 | 74.6 | +11.3% |
| + DINO attention | 84.3 | 86.2 | 76.3 | DINO supplements spatial relations |
| + Full SGC | **87.9** | **89.1** | **78.7** | Mask filtering yields further gains |

Loss function ablation (VOC train mIoU):

| Configuration | mIoU | Note |
|------|------|------|
| CLIP baseline | 58.6 | Original |
| + Direct feature fine-tuning | 53.5 | −5.1%, degrades CLIP capability |
| + Intra-modal contrastive | 57.8 | Only −0.8%, but limited effectiveness |
| **+ Cross-modal contrastive** | **63.3** | +4.7%, effectively bridges modality gap |

### Key Findings

1. Cross-modal contrastive learning is more effective than intra-modal contrastive learning and direct fine-tuning, validating the modality gap as the core bottleneck.
2. The fusion of CLIP and DINO attention exhibits strong complementarity: CLIP provides semantic guidance while DINO contributes spatial priors.
3. SGC's superpixel mask filtering effectively suppresses spurious background responses, improving both precision and recall simultaneously.
4. SSR's mIoU reaches 97.4% of fully supervised methods, demonstrating the substantial potential of weakly supervised segmentation.

## Highlights & Insights

1. **Incisive problem decomposition**: The difficulties of CLIP-based WSSS are explicitly attributed to cross-modal semantic misalignment at the semantic level and affinity noise at the spatial level, enabling targeted solutions.
2. **Elegant cross-modal prototype design**: Prototypes are constructed in the projection space rather than the original feature space, preserving CLIP's discriminative capability while reducing computational cost.
3. **SLIC over SAM**: Choosing lightweight SLIC over heavyweight SAM for spatial prior extraction reflects pragmatic engineering judgment.
4. **CLIP + DINO fusion**: Leveraging the complementarity of two pretrained models (global semantics vs. local spatial structure) exemplifies effective multi-model collaboration.

## Limitations & Future Work

1. Superpixel parameters in SGC (e.g., SLIC segment count and compactness) may require tuning across different datasets.
2. The prototype update frequency (every 5,000 steps) is relatively coarse; smoother update strategies such as exponential moving average could be considered.
3. Reliance on DINO attention to supplement spatial information introduces an additional model dependency.
4. The 50.6% mIoU on COCO leaves considerable room for improvement; the complexity of multi-category scenarios warrants further investigation.

## Related Work & Insights

- **VPL** (CVPR 2025): Learns category-specific prototypes in the visual space as an alternative to text prototypes, complementing the CMPA perspective.
- **ExCEL** (CVPR 2025): Employs LLMs to generate fine-grained category descriptions for enriched text prompts, contrasting with CMPA's feature-space alignment strategy.
- **CLIP-ES** (CVPR 2023): SSR's text prompt design references the background category construction approach introduced in CLIP-ES.
- The cross-modal contrastive learning paradigm of CMPA is transferable to other downstream tasks requiring vision-language alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The dual-level rectification framework is well-motivated, though contrastive learning and superpixel guidance are not entirely novel concepts.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Evaluated on both VOC and COCO with multi-dimensional metrics and comprehensive ablations.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear figures and well-articulated motivation.)
- **Value**: ⭐⭐⭐⭐ (Achieves near fully supervised segmentation performance under weak supervision, with significant practical implications.)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](../../ICCV2025/segmentation/know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[CVPR 2026\] Weakly-Supervised Referring Video Object Segmentation through Text Supervision](../../CVPR2026/segmentation/wsrvos_weakly_supervised_rvos.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](../../CVPR2026/segmentation/fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](../../CVPR2026/segmentation/clip_shortsighted_beyond_first_sentence.md)

<!-- RELATED:END -->
