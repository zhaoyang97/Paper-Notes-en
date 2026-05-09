---
title: >-
  [Paper Note] Object-level Correlation for Few-Shot Segmentation
description: >-
  [ICCV 2025][Segmentation][few-shot segmentation] OCNet is proposed to construct **object-level** (rather than image-level) support-query correlations by emulating biological visual processes. It first mines generic objects in the query image and then identifies the target object among them, effectively suppressing irrelevant object noise in the background.
tags:
  - ICCV 2025
  - Segmentation
  - few-shot segmentation
  - object-level correlation
  - prototype learning
  - optimal transport
  - hard pixel noise
date: 2026-05-08
content_hash: 18275f0ea372f9d9
---

# Object-level Correlation for Few-Shot Segmentation

**Conference**: ICCV 2025
**arXiv**: [2509.07917](https://arxiv.org/abs/2509.07917)
**Code**: N/A
**Area**: Image Segmentation
**Keywords**: few-shot segmentation, object-level correlation, prototype learning, optimal transport, hard pixel noise

## TL;DR

OCNet is proposed to construct **object-level** (rather than image-level) support-query correlations by emulating biological visual processes. It first mines generic objects in the query image and then identifies the target object among them, effectively suppressing irrelevant object noise in the background.

## Background & Motivation

The core challenge of few-shot semantic segmentation (FSS) lies in establishing correlations between the support target and the query image. Existing methods primarily build **image-level correlations** (support target ↔ entire query image), which suffer from the following issues:

1. **Hard pixel noise**: correlations include irrelevant background objects (e.g., true background elements, base-class objects, irrelevant novel-class objects).
2. Post-processing methods such as BAM and ABCB attempt to eliminate some noise, but still fail to handle **irrelevant novel-class objects** (e.g., when both a dog and a person appear in the query image but only the dog needs to be segmented).
3. When multiple novel-class objects co-occur, image-level correlations struggle to accurately identify the target.

**Biological vision inspiration**: The human visual system first computes global saliency in a pre-attentive manner (locating generic objects) and then selects the target based on task cues. Target recognition is more effective when performed over generic objects than over the entire image.

## Method

### Overall Architecture

OCNet consists of two core modules:
1. **GOMM** (General Object Mining Module): mines generic object features from the query image.
2. **CCM** (Correlation Construction Module): constructs object-level correlations between the support target and the query generic objects.

Pipeline: a pretrained backbone extracts features → GOMM generates generic object features $F_g$ → CCM leverages support prototypes and $F_g$ to build object-level correlation $F_c$ → FPN decoder produces predictions.

### Key Designs

1. **GOMM – General Object Mining Module**:

    - **Generic object mask generation**: Since query ground truth is unavailable, CAM is used to obtain raw generic object masks, fused with a cosine similarity prior between high-level support and query features, and thresholded at $\tau=0.6$:
    $M_g = \mathbb{1}_\tau(\text{Max}(\text{Cosine}(F_q^h, F_s^h) \oplus \text{CAM}(F_q^h)))$
    - **Initial generic object features**: Randomly initialized generic object prototypes $P_g \in \mathbb{R}^{N_g \times C}$ are used to compute cosine similarity assignments with the query features; the results are concatenated and passed through a 1×1 convolution to produce $F_{ig}$.
    - **Information completion**: Cross-attention fuses $F_{ig}$ and $F_q$: $F_g = \text{Atten}(F_q, F_{ig}, F_{ig}) + F_q$
    - **Design motivation**: Although the generic object masks are imperfect, moderate incompleteness is beneficial for the generalization and reconstruction capacity of the prototypes.

2. **CCM – Correlation Construction Module**:

    - **Support prototype extraction**: Multi-frequency pooling (MFP) generates prototypes $P_s \in \mathbb{R}^{L \times C}$ ($L=49$) from support features.
    - **Foreground/background prototype selection**: Euclidean distance is used to compare prototype activation masks $M_{sp}$ with the ground-truth mask $M_s$; TopK indices $ID_t$ identify foreground prototypes and LowK indices $ID_l$ identify background prototypes.
    - **Optimal transport assignment**: Prototype assignment is formulated as an OT problem, solved via the Sinkhorn algorithm ($\epsilon=0.05$) to obtain the optimal transport matrix $T^*$ and the prototype assignment mask $M_{pa}$.
    - **Correlation construction**: The assignment mask supervises prototype allocation; matrix multiplication fuses support and query information to produce the object-level correlation $F_c = \text{Alloc}(P_q, \text{Argmax}(\hat{M}_{pa})) \oplus F_g$.
    - **Design motivation**: Foreground prototypes capture target information, while background prototypes actively suppress noisy pixels—a role that prior methods overlooked.

3. **Dual foreground + background prototype mechanism**:

    - Unlike prior methods such as FPTrans that rely solely on foreground prototypes, CCM leverages both foreground and background prototypes.
    - Foreground prototypes activate target regions; background prototypes suppress hard pixel noise.
    - Both are unified under the optimal transport framework via the assignment mask.
    - **Design motivation**: Noise suppression is equally important as target enhancement.

### Loss & Training

The total loss consists of three terms: $\mathcal{L}_f = \mathcal{L}_t + \mathcal{L}_g + \mathcal{L}_p$
- $\mathcal{L}_t = \text{CE}(\hat{M}_q, M_q)$: target segmentation loss
- $\mathcal{L}_g = \text{CE}(\hat{M}_g, M_g)$: generic object segmentation loss
- $\mathcal{L}_p = \text{CE}(\hat{M}_{pa}, M_{pa})$: prototype assignment loss

Training configuration: SGD optimizer, lr=0.005, batch size=4; 200 epochs on PASCAL-5^i, 75 epochs on COCO-20^i; images cropped to 473×473 (PASCAL) or 641×641 (COCO).

## Key Experimental Results

### Main Results

**PASCAL-5^i 1-shot/5-shot (ResNet-50)**:

| Method | 1-shot Mean mIoU | 1-shot FB-IoU | 5-shot Mean mIoU | 5-shot FB-IoU |
|------|-----------------|---------------|-----------------|---------------|
| BAM (CVPR'22) | 67.8 | 79.7 | 70.9 | 82.2 |
| AENet (ECCV'24) | 69.8 | 80.8 | 74.1 | 84.5 |
| ABCB (CVPR'24) | 70.6 | - | 73.6 | - |
| HMNet (NeurIPS'24) | 70.4 | 81.6 | 74.1 | 84.4 |
| **OCNet** | **71.4** | **82.2** | **74.5** | **84.7** |

**COCO-20^i 1-shot/5-shot (ResNet-50)**:

| Method | 1-shot Mean mIoU | 1-shot FB-IoU | 5-shot Mean mIoU | 5-shot FB-IoU |
|------|-----------------|---------------|-----------------|---------------|
| AENet (ECCV'24) | 49.4 | 73.6 | 56.7 | 76.5 |
| ABCB (CVPR'24) | 50.0 | - | 55.1 | - |
| **OCNet** | **51.5** | **73.7** | **57.0** | **76.8** |

### Ablation Study

**GOMM and CCM module ablation (PASCAL-5^i, 1-shot, ResNet-50)**:

| GOMM | CCM | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
|------|-----|--------|--------|--------|--------|------|
| ✗ | ✗ | 67.5 | 73.4 | 66.5 | 61.6 | 67.3 |
| ✓ | ✗ | 69.9 | 74.2 | 68.3 | 63.9 | 69.1 |
| ✗ | ✓ | 71.9 | 74.7 | 69.8 | 63.0 | 69.9 |
| ✓ | ✓ | **73.5** | **75.9** | **71.1** | **64.9** | **71.4** |

- GOMM alone contributes +1.8% mIoU.
- CCM alone contributes +2.6% mIoU.
- Their combination yields +4.1% mIoU, demonstrating complementary effectiveness.

### Key Findings

- Object-level correlation consistently outperforms image-level correlation across all settings, validating the "locate generic objects first, then identify the target" strategy.
- The introduction of background prototypes is crucial for suppressing hard pixel noise—a factor neglected by prior methods.
- Although the generic object masks are imperfect, moderate incompleteness benefits prototype generalization.
- OCNet's advantages become more pronounced on more challenging datasets such as COCO-20^i.
- Consistent improvements are observed with both VGG-16 and ResNet-50 backbones.

## Highlights & Insights

1. **Biological vision inspiration**: The method emulates the two-stage human visual process of "saliency → target selection," translating an abstract cognitive mechanism into computable modules.
2. **Paradigm shift from image-level to object-level correlation**: Rather than matching the entire query image with the support, OCNet first extracts generic objects and then establishes precise correspondences.
3. **Dual foreground + background prototype mechanism**: Background prototypes actively suppress noise rather than relying solely on passive filtering.
4. **Optimal transport for assignment modeling**: Prototype-to-pixel assignment is formulated as an OT problem, enabling globally optimal allocation.

## Limitations & Future Work

- The quality of CAM-generated generic object masks is inconsistent and may occasionally miss important targets.
- Optimal transport solving increases computational overhead, requiring a trade-off in Sinkhorn iteration count.
- Evaluation is limited to PASCAL-5^i and COCO-20^i; performance in other domains (e.g., medical imaging, remote sensing) remains untested.
- When the query image contains only a single object, the advantages of object-level correlation may be less pronounced than in multi-object scenarios.
- Comparisons with methods based on large-scale pretrained models (e.g., SAM) are not included.

## Related Work & Insights

- The concept of object-level correlation can be extended to support-query matching in other dense prediction tasks.
- The two-stage strategy of "segment generic objects first, then identify the target" may be particularly valuable in open-world settings.
- Modeling prototype assignment via OT is a direction worthy of further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The shift from image-level to object-level correlation is creative, and the biological vision motivation is well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dual datasets, dual backbones, comprehensive ablations, and clear qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, the method is described in detail, and the figures are informative.
- **Value**: ⭐⭐⭐⭐ Offers a new perspective for FSS, though the performance gains are moderate (~1–2% mIoU).

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](../../NeurIPS2025/segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)
- [\[ICCV 2025\] LayerAnimate: Layer-level Control for Animation](layeranimate_layer-level_control_for_animation.md)
- [\[ICCV 2025\] ZIM: Zero-Shot Image Matting for Anything](zim_zero-shot_image_matting_for_anything.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)

<!-- RELATED:END -->
