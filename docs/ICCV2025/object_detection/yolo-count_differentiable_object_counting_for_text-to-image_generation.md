---
title: >-
  [Paper Note] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation
description: >-
  [ICCV 2025][Object Detection][Object Counting] This paper proposes YOLO-Count, a fully differentiable open-vocabulary object counting model built upon the YOLO architecture. Through an innovative cardinality map regressi…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Object Counting"
  - "Differentiable"
  - "Text-to-Image Generation"
  - "Cardinality Map"
  - "Open Vocabulary"
date: 2026-05-08
content_hash: 516765b70cbf479e
---

# YOLO-Count: Differentiable Object Counting for Text-to-Image Generation

**Conference**: ICCV 2025
**arXiv**: [2508.00728](https://arxiv.org/abs/2508.00728)  
**Code**: None  
**Area**: Object Detection
**Keywords**: Object Counting, Differentiable, Text-to-Image Generation, Cardinality Map, Open Vocabulary

## TL;DR

This paper proposes YOLO-Count, a fully differentiable open-vocabulary object counting model built upon the YOLO architecture. Through an innovative cardinality map regression target and a hybrid strong-weak supervised training strategy, YOLO-Count achieves state-of-the-art performance on both general object counting and quantity-controlled text-to-image generation.

## Background & Motivation

Text-to-image (T2I) generation models have achieved remarkable progress in synthesizing high-quality images, yet they still face significant challenges in **precisely controlling the number of generated objects**. Unlike local attributes such as color and texture, object quantity is a **global constraint** that requires the model to establish a numerical correspondence between language tokens and generated instances.

Limitations of existing approaches:

**Detection-based counting models** (e.g., CountGD, DAVE): enumerate discrete counts via object detection followed by threshold filtering; **their outputs are non-differentiable**, precluding direct use in gradient-based T2I generation control.

**Density map regression models** (e.g., CLIP-Count, VLCounter): although differentiable, density maps suffer from inherent ambiguity—the center position and radius of the Gaussian kernel are arbitrarily chosen, causing systematic over-counting for large objects.

**T2I quantity control methods** (e.g., BoxDiff): primarily modulate cross-attention, but attention mechanisms are better suited to distinguishing categories than differentiating multiple instances of the same category.

**An ideal T2I counting guidance model should satisfy four properties**:
- Fully differentiable with respect to the input image
- Open-vocabulary capability
- Cross-scale generalization
- Computational efficiency

## Method

### Overall Architecture

YOLO-Count is built upon the YOLO-World architecture and comprises three core components: a visual backbone, a Vision-Language Path Aggregation Network (VLPAN), and prediction heads (a classification head and a cardinality regression head). Training follows a two-stage hybrid strong-weak supervision scheme: strong-supervised pre-training on LVIS instance segmentation data, followed by weakly supervised fine-tuning on the FSC147 counting dataset.

### Key Designs

1. **Cardinality Map Regression**:

    - Function: Replaces conventional density maps, providing an unambiguous regression target for counting.
    - Mechanism: Given the binary mask $M_i$ of the $i$-th instance (with area $N_i$), the value 1 is uniformly distributed over all pixels covered by the object:
    $y_{\text{pixel cardinality}} = \sum_{i=1}^{K} \frac{1}{N_i} M_i$
      This is then downsampled to the grid level:
    $y_{\text{car}}(u,v) = \sum_{(i,j) \in \Omega_{u,v}} y_{\text{pixel cardinality}}(i,j)$
      The sum of the cardinality map strictly equals the object count: $\sum_{u,v} y_{\text{car}}(u,v) = Q$
    - Design Motivation: Density maps concentrate values at Gaussian kernel centers, introducing two sources of ambiguity—the kernel center can be placed arbitrarily within the object, and the kernel radius is chosen heuristically. The cardinality map eliminates these ambiguities by uniformly covering the full spatial extent of each object, yielding greater robustness across varying object sizes and shapes. Experiments confirm that density map methods tend to over-count as object size increases.

2. **Representation Alignment**:

    - Function: Aligns visual and textual representations via a contrastive learning branch to ensure the model can effectively localize objects of specified categories.
    - Mechanism: The problem is formulated as a binary classification task, where each pixel is classified as belonging or not belonging to the target category:
    $\mathcal{L}_{\text{cls}} = \text{BCELoss}(\hat{y}_{\text{cls}}, y_{\text{cls}})$
      Prediction probabilities are computed via the inner product of visual features $o_{\text{cls}}$ and CLIP text embeddings $f_T$, followed by sigmoid, analogous to SigLIP.
    - Design Motivation: Ensures the counting model not only counts but also accurately discriminates object categories, which is critical in open-vocabulary scenarios.

3. **Hybrid Strong-Weak Supervised Training**:

    - Function: Jointly leverages large-scale instance segmentation datasets and sparsely annotated counting datasets.
    - Mechanism:
        - **Strong-supervised pre-training** (LVIS): Cardinality maps and classification masks are constructed from precise instance segmentation masks.
       $$\mathcal{L}_{\text{total}}^{\text{strong}} = \alpha_1 \mathcal{L}_{\text{cnt}}^{\text{strong}} + \beta_1 \mathcal{L}_{\text{cls}}^{\text{strong}}$$
        - **Weakly supervised fine-tuning** (FSC147): Sparse point annotations are used, with positive samples from annotated points and negative samples from manually annotated background points.
       $$\mathcal{L}_{\text{cnt}}^{\text{weak}} = |(\sum_p \hat{y}_{\text{cnt}}(p)) - K|$$
       A proportion $\gamma$ of LVIS data is retained during fine-tuning to preserve open-vocabulary capability.
    - Design Motivation: Counting datasets are small-scale and category-limited (e.g., FSC147 contains only 3,659 training images), while instance segmentation datasets (LVIS, 1,203 categories) provide rich, precise annotations. The two sources are complementary and together improve generalization.

### Loss & Training

- Strong-supervised pre-training: 250 epochs on LVIS; $\alpha_1=1.0,\ \beta_1=0.1$
- Weakly supervised fine-tuning: up to 500 epochs on FSC147; $\gamma=0.05$ (retaining 5% LVIS data)
- Backbone initialized from YOLOv8l weights; CLIP text encoder frozen
- Differentiated learning rates: backbone $5\times10^{-9}$, new modules $1\times10^{-5}$
- T2I quantity control: based on textual inversion, iteratively optimizing the count token embedding for up to 150 steps

## Key Experimental Results

### Main Results

| Model | FSC-Test MAE↓ | FSC-Test RMSE↓ | LVIS MAE↓ | OpenImg7-New MAE↓ | Obj365-New MAE↓ |
|------|--------------|----------------|-----------|-------------------|-----------------|
| CountGD (non-differentiable) | 12.98 | 98.35 | 4.84 | 6.09 | 3.53 |
| CLIP-Count (differentiable) | 17.78 | 106.62 | 10.81 | 14.01 | 15.48 |
| VLCounter (differentiable) | 17.05 | 106.16 | 8.94 | 15.32 | 18.08 |
| **YOLO-Count (differentiable)** | **14.80** | **96.14** | **1.65** | **3.72** | **3.28** |

YOLO-Count achieves comprehensive state-of-the-art performance among differentiable models and substantially outperforms competitors on open-vocabulary counting across LVIS, OpenImg7, and Obj365. With fewer parameters, it even surpasses the non-differentiable CountGD on open-vocabulary benchmarks.

### Ablation Study

| Configuration | FSC-Test MAE↓ | FSC-Test RMSE↓ | Notes |
|------|--------------|----------------|------|
| Full YOLO-Count | **14.80** | **96.14** | Baseline |
| w/o pre-training (no LVIS) | 18.42 | 111.45 | MAE +3.62 |
| w/o weak supervision (no FSC147) | 43.91 | 150.40 | Severe degradation |
| w/o cardinality map (density map) | 16.71 | 107.24 | MAE +1.91 |
| w/o alignment (no classification branch) | 17.01 | 110.41 | MAE +2.21 |
| w/o additional VLPAN | 16.54 | 106.32 | MAE +1.74 |

Each component contributes meaningfully. Weakly supervised fine-tuning is the most critical component—removing it causes MAE to increase from 14.80 to 43.91. The cardinality map reduces MAE by 1.91 compared to density maps.

### Key Findings

1. **Scale bias of density maps**: Density map methods systematically over-count as object size increases, since Gaussian kernels for large objects may span multiple grid cells, causing the sum to deviate from the ground-truth count. The cardinality map eliminates this bias and achieves stability comparable to detection-based methods.
2. **T2I quantity control**: On the LargeGen benchmark from FSC147 (target counts 25–100), YOLO-Count-guided generation substantially outperforms CountGD (non-differentiable proxy loss) and CLIP-Count (density map) in count accuracy, with particularly pronounced advantages at large counts (75, 100).
3. Although CountGD achieves high counting accuracy, its non-differentiable output necessitates a proxy loss for T2I guidance, which ultimately degrades generation quality.
4. Retaining only 5% of LVIS data during hybrid training is sufficient to effectively preserve open-vocabulary capability.

## Highlights & Insights

- **The cardinality map is a concise yet effective innovation**: uniformly distributing values over instance masks instead of placing Gaussian kernels fundamentally eliminates the ambiguities inherent in density maps—a simple idea with significant empirical impact.
- **Bridging counting and generation**: This work is the first to systematically connect open-vocabulary object counting with quantity control in T2I generation, leveraging a fully differentiable architecture to enable gradient pass-through.
- **Practical hybrid training strategy**: Pre-training on existing instance segmentation annotations and requiring only minimal manual background point annotation for fine-tuning substantially reduces the cost of constructing counting datasets.
- **New open-vocabulary counting benchmarks**: Two new evaluation benchmarks, OpenImg7-New and Obj365-New, are introduced.

## Limitations & Future Work

1. Cardinality map construction requires instance segmentation masks, constraining the data sources available for strong-supervised pre-training.
2. The weakly supervised stage requires manual annotation of background negative points (~5 seconds per image), which, while efficient, still incurs a human annotation cost.
3. T2I quantity control employs single-step inference with SDXL-Turbo, which limits generation quality.
4. On the conventional FSC147 benchmark, performance remains slightly below non-differentiable CountGD (MAE 14.80 vs. 12.98), indicating a modest accuracy trade-off for differentiability.
5. The current model supports single-category counting only; simultaneous multi-category counting remains an open direction.

## Related Work & Insights

- Built upon YOLO-World, inheriting its vision-language fusion capability while replacing detection outputs with counting regression.
- The analysis of limitations in density map methods (CLIP-Count, VLCounter) contributes new understanding to the counting literature.
- The textual inversion + differentiable guidance framework for T2I control is generalizable to other global constraints such as scene layout and style consistency.

## Rating

- Novelty: ⭐⭐⭐⭐ The cardinality map concept is novel, and the connection between counting and T2I generation is meaningful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five counting benchmarks, T2I control evaluation, comprehensive ablation, and scale bias analysis.
- Writing Quality: ⭐⭐⭐⭐ Method presentation is clear, motivations are well-justified, and visualizations are informative.
- Value: ⭐⭐⭐⭐ Advances both open-vocabulary counting and controllable generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer](../../AAAI2026/object_detection/anostyler_text-driven_localized_anomaly_generation_via_light.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[CVPR 2026\] Does YOLO Really Need to See Every Training Image in Every Epoch?](../../CVPR2026/object_detection/does_yolo_really_need_to_see_every_training_image_in_every_epoch.md)
- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[ICCV 2025\] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)

</div>

<!-- RELATED:END -->
