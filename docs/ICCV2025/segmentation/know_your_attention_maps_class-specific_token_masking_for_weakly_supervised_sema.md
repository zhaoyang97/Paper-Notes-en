---
title: >-
  [Paper Note] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation
description: >-
  [ICCV 2025][Segmentation][Weakly Supervised Semantic Segmentation] This paper proposes an end-to-end weakly supervised semantic segmentation method that introduces multiple [CLS] tokens (one per class) into a ViT…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Weakly Supervised Semantic Segmentation"
  - "Vision Transformer"
  - "Attention Maps"
  - "CLS Token"
  - "Attention Head Pruning"
date: 2026-05-08
content_hash: a3f44e7c58d81e49
---

# Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.06848](https://arxiv.org/abs/2507.06848)  
**Code**: [github.com/HSG-AIML/TokenMasking-WSSS](https://github.com/HSG-AIML/TokenMasking-WSSS)  
**Area**: Semantic Segmentation (Weakly Supervised)
**Keywords**: Weakly Supervised Semantic Segmentation, Vision Transformer, Attention Maps, CLS Token, Attention Head Pruning

## TL;DR

This paper proposes an end-to-end weakly supervised semantic segmentation method that introduces multiple [CLS] tokens (one per class) into a ViT, applies random masking to [CLS] token output embeddings, and prunes redundant attention heads. Class-specific pseudo segmentation masks are generated directly from self-attention maps without any additional CAM module.

## Background & Motivation

- **Weakly Supervised Semantic Segmentation (WSSS)** aims to achieve pixel-level segmentation using only image-level labels, reducing the need for fine-grained annotations.
- Conventional methods rely on **Class Activation Maps (CAM)** to highlight discriminative regions and generate pseudo segmentation masks; however, CAM suffers from coarse localization, tends to focus only on the most salient regions, and requires additional modules.
- The self-attention mechanism of ViTs provides natural interpretability, but two key challenges remain:
    - It is not possible to assign specific classes to individual attention heads.
    - There is no guarantee that different [CLS] tokens are hard-assigned to their corresponding classes.
- This paper aims to bridge the gap between ViT capabilities and the need for class-specific feature embeddings.

## Method

### Overall Architecture

The input image is split into patches and projected into a token sequence. $C$ [CLS] tokens (where $C$ is the number of classes) and one [REG] token are prepended to the sequence. After passing through the Transformer encoder, the self-attention maps of each [CLS] token are used to generate pseudo segmentation masks.

### Key Designs

1. **Multiple [CLS] Token Design**: The standard single [CLS] token in ViT is extended to $C$ tokens, each corresponding to one semantic class. The extended embedding sequence is $\mathbf{z}_0 = [\text{[CLS]}_1; \text{[CLS]}_2; \ldots; \text{[CLS]}_C; \mathbf{z}_0; \text{[REG]}]$. The [REG] token, inspired by Darcet et al., captures global context information and prevents the self-attention of [CLS] tokens from being "contaminated."

2. **Random Masking of [CLS] Tokens**: During training, 50% of the [CLS] tokens that do not correspond to the current image's labels are randomly selected and have their output embeddings masked (set to zero). The core idea is that after masking, the model must rely on the remaining available tokens to make class decisions, thereby forcing each [CLS] token to learn the correct class assignment. The masking function is defined as: $m(i) = 0$ if $i \in \mathcal{Y}$ (the ground-truth label set), otherwise set to 1 with 50% probability.

3. **Attention Head Pruning**: A learnable gate scalar $g_i$ is introduced for each attention head, modifying the MSA as $\text{MSA}(Q,K,V) = \text{concat}_i(g_i \cdot \text{head}_i)W^O$. Differentiable pruning is achieved via stochastic relaxation of the $L_0$ norm using the Hard Concrete distribution. The regularization loss $\mathcal{L}_{\text{reg}} = \sum_i(1 - P(g_i=0|\phi_i))$, with $\lambda=0.01$, prunes approximately two-thirds of the attention heads, removing redundant and noisy heads and making the remaining heads more interpretable.

### Loss & Training

- Total loss: $\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{reg}}$
- $\mathcal{L}_{\text{cls}}$ is the binary cross-entropy classification loss.
- $\lambda = 0.01$ controls the degree of pruning.
- At inference: the self-attention maps of [CLS] tokens corresponding to predicted classes are extracted, reshaped to the image spatial dimensions, binarized, and merged in order of increasing class probability. Unassigned pixels are filled using the most common value in their neighborhood.

## Key Experimental Results

### Main Results

**Pseudo Mask Quality (Pascal VOC 2012, mIoU %)**:

| Method | Type | Backbone | train | val |
|--------|------|----------|-------|-----|
| ViT-PCM + CRF (ECCV'22) | Multi-stage | ViT-B† | 71.4 | 69.3 |
| ReCAM (CVPR'22) | Multi-stage | ResNet50 | 70.5 | - |
| I/C-CTI (CVPR'24) | Multi-stage | DeiT-S | 73.7 | - |
| AFA (CVPR'22) | Single-stage | MiT-B1 | 68.7 | 66.5 |
| ToCo (CVPR'23) | Single-stage | ViT-B | 72.2 | 70.5 |
| DuPL (CVPR'24) | Single-stage | ViT-B | 75.1 | 73.5 |
| **Ours** | Single-stage | ViT-B | 74.5 | **73.7** |

**Final Segmentation Results (mIoU %)**:

| Method | Backbone | VOC val | VOC test | COCO val |
|--------|----------|---------|----------|----------|
| ReCAM (CVPR'22) | DL-V2 | 68.4 | 68.2 | 45.0 |
| I/C-CTI (CVPR'24) | ResNet38 | 74.1 | 73.2 | 45.4 |
| AFA (CVPR'22) | MiT-B1 | 66.0 | 66.3 | 38.9 |
| ToCo (CVPR'23) | ViT-B | 69.8 | 70.5 | 41.3 |
| DuPL (CVPR'24) | ViT-B | 72.2 | 71.6 | 43.6 |
| **Ours** | ViT-B | 72.7 | **73.5** | 43.2 |

### Ablation Study

**Effect of Each Component (mIoU %)**:

| Component | MS COCO | VOC |
|-----------|---------|-----|
| w/o Random Masking | 41.9 | 71.6 |
| w/ Random Masking | **43.2** | **72.7** |
| w/o [REG] Token | 42.8 | 72.3 |
| w/ [REG] Token | **43.2** | **72.7** |
| w/o Attention Head Pruning | 41.7 | 72.0 |
| w/ Attention Head Pruning | **43.2** | **72.7** |

**Sensitivity Analysis of Masking Ratio**: Masking ratios from 0% to 100% are evaluated; performance peaks around 50%, after which it plateaus or slightly decreases.

### Key Findings

- State-of-the-art weakly supervised performance is achieved on three domain-specific datasets (DFC2020 remote sensing, EndoTect medical, ADE20K scene understanding); on DFC2020, the proposed method even surpasses fully supervised approaches (mIoU 67.2 vs. 53.1).
- Random masking is the most critical component: without masking, attention shapes are accurate but class assignments are incorrect.
- The [REG] token improves class boundary clarity, particularly between categories such as "sky" and "building."
- Attention head pruning yields smoother pseudo masks with less noise.

## Highlights & Insights

- The core idea is elegant and concise: an end-to-end WSSS framework is realized through multiple [CLS] tokens, random masking, and pruning, without requiring additional modules.
- The random masking strategy is cleverly designed: by an "exclusion" mechanism, each [CLS] token is forced to specialize in a specific class.
- The method is particularly effective in annotation-scarce specialized domains (remote sensing and medical imaging), reducing dependence on large-scale labeled data.
- Surpassing fully supervised methods on the DFC2020 remote sensing dataset demonstrates that weakly supervised learning with effective attention utilization can outperform coarsely supervised approaches.

## Limitations & Future Work

- As the number of classes grows, the number of [CLS] tokens increases linearly, raising model parameters and computational complexity; **scalability is thus limited**.
- Efficiency in large-scale category settings such as ADE20K (150 classes) remains an open challenge.
- Dynamic token allocation strategies could be explored to reduce unnecessary computation.
- Integration with foundation models such as SAM has not been investigated and may further improve quality.

## Related Work & Insights

- Similar to MCTformer (CVPR'22) in using multiple class tokens, the proposed masking strategy achieves more reliable class assignment.
- The [REG] token design is inspired by the Vision Transformer Register work of Darcet et al.
- DINO's self-supervised ViT also demonstrates the interpretability of self-attention; this work builds upon that foundation by imposing constraints to achieve class-specific attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of multiple CLS tokens, random masking, and pruning is novel and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five datasets (standard benchmarks and specialized domains) with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with intuitive figures.
- **Value**: ⭐⭐⭐⭐ Practically valuable for weakly supervised segmentation, especially in annotation-scarce specialized domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation](../../AAAI2026/segmentation/ssr_semantic_and_spatial_rectification_for_clip-based_weakly_supervised_segmenta.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction](conformalsam_unlocking_the_potential_of_foundational_segmentation_models_in_semi.md)
- [\[ICCV 2025\] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup](dictas_a_framework_for_class-generalizable_few-shot_anomaly_segmentation_via_dic.md)
- [\[ICCV 2025\] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP](know_no_better_a_data-driven_approach_for_enhancing_negation_awareness_in_clip.md)

</div>

<!-- RELATED:END -->
