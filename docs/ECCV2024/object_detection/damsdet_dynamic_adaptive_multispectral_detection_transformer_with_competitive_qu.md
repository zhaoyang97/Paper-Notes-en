---
title: >-
  [Paper Note] DAMSDet: Dynamic Adaptive Multispectral Detection Transformer
description: >-
  [ECCV 2024][Object Detection][Infrared-Visible Detection] DAMSDet proposes a dynamic adaptive infrared-visible object detection method based on the DETR architecture. By utilizing Modality Competitive Query Selection (dynamically selecting the dominant modality feature as the initial query for each object) and Multispectral Deformable Cross-Attention (adaptively sampling and aggregating bi-modal features across multiple semantic levels), it simultaneously addresses the dual c…
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Infrared-Visible Detection"
  - "Multispectral Fusion"
  - "DETR"
  - "Modality Competitive Selection"
  - "Deformable Cross-Attention"
date: 2026-05-08
content_hash: 88ea3e007d850215
---

# DAMSDet: Dynamic Adaptive Multispectral Detection Transformer

**Conference**: ECCV 2024  
**arXiv**: [2403.00326](https://arxiv.org/abs/2403.00326)  
**Code**: [GitHub](https://github.com/gjj45/DAMSDet)  
**Area**: Object Detection  
**Keywords**: Infrared-Visible Detection, Multispectral Fusion, DETR, Modality Competitive Selection, Deformable Cross-Attention

## TL;DR

DAMSDet proposes a dynamic adaptive infrared-visible object detection method based on the DETR architecture. By utilizing Modality Competitive Query Selection (dynamically selecting the dominant modality feature as the initial query for each object) and Multispectral Deformable Cross-Attention (adaptively sampling and aggregating bi-modal features across multiple semantic levels), it simultaneously addresses the dual challenges of complementary information fusion and modality misalignment, significantly outperforming the state-of-the-art (SOTA) on four public datasets.

## Background & Motivation

**Background**: Infrared-visible object detection achieves all-weather robust detection by fusing complementary information from dual modalities. Infrared imaging is unaffected by illumination and smoke but lacks texture details; visible light provides rich textures and colors but is constrained by low-light environments.

**Limitations of Prior Work**:
   - **Difficulty in Complementary Feature Fusion**: The complementary characteristics of different objects in the same scene vary significantly; some objects completely disappear in visible light (due to smoke occlusion), while others have partial information in both modalities. Global weights or region-level fusion methods lack sufficient granularity.
   - **Modality Misalignment**: Infrared and visible images often have registration offsets; even after manual registration, objects still exhibit displacement. AR-CNN requires extra paired bounding box annotations to learn offsets, which is highly expensive.
   - **Coarse Existing Fusion Methods**: One-step fusion or whole-image feature fusion struggles to fully exploit complementary information in complex scenes.

**Mechanism**: Drawing inspiration from the human observation pattern—focusing on salient objects in each modality first, and then progressively aggregating key information from both modalities. A DETR cascaded decoder structure is used for layer-by-layer refinement.

## Method

### Overall Architecture

DAMSDet consists of four main components:
1. **Bi-modal Specific CNN Backbones**: Extract infrared and visible features respectively.
2. **Bi-modal Specific Efficient Encoders**: Reference the efficient encoder design of RT-DETR, combining Transformers and CNNs to reduce computational complexity.
3. **Modality Competitive Query Selection (MCQS)**: Competitively selects salient modality features from the bi-modal encoded features to serve as initial queries.
4. **Multispectral Transformer Decoder**: Progressively refines queries across multiple semantic levels using multispectral deformable cross-attention.

### Key Designs

**1. Modality Competitive Query Selection (MCQS)**

- Concatenates the bi-modal encoded feature sequences and projects them through a linear layer to obtain a score for each feature point.
- Selects the Top-$K$ highest-scoring features as initial queries: $z = \text{Top-}K(\text{Linear}(\text{concat}(I, V)))$
- Each query originates from a specific modality (infrared or visible), representing an object instance in that modality.
- **Competitive Mechanism**: Automatically selects the "dominant modality" with stronger signals for each object, preventing the introduction of noisy information at an early stage.
- Works in conjunction with IoU-aware classification loss to further improve selection quality.
- Redundant queries (pointing to the same object across both modalities) are naturally eliminated through the one-to-one matching and self-attention mechanisms of DETR.
- Introduces Noise Query Learning during training to assist in learning optimal modality matching.
- Visual validation: Different objects are indeed selected by different dominant modalities, consistent with physical intuition.

**2. Multispectral Transformer Decoder**

Each decoder layer consists of:
- **Multi-head Self-Attention**: Discovers contextual information and reduces redundancy.
- **Multispectral Deformable Cross-Attention (MDCA)**: The core fusion module.
- **4D Anchor Constraints**: Uses 4D reference points $(x,y,w,h)$ to constrain the sampling range, iteratively refining layer-by-layer: $b_{q}^{d} = \sigma(MLP^d(z_q^d) + \sigma^{-1}(b_q^{d-1}))$

**3. Multispectral Deformable Cross-Attention (MDCA)**

- Extends the deformable attention of Deformable DETR into a multi-modal format.
- Adaptively performs sparse sampling and weighted aggregation on bi-modal multi-semantic level feature maps.
- $m \in \{1,2\}$ represent visible and infrared modalities respectively, $l$ indexes semantic levels, and $k$ indexes sampling points.
- Each modality **independently predicts sampling offsets**, naturally adapting to modality misalignment.
- Attention weights are normalized across both modalities, multiple semantic levels, and multiple sampling points: $\sum_m \sum_l \sum_k A_{mhlqk} = 1$
- Offsets are constrained within the reference bounding box, focusing on the information surrounding the object.
- **Visual Analysis**:
    - Deep decoder layers tend to focus on low-level semantic features of the infrared modality (basic outlines) and high-level semantic features of the visible modality (category relationships).
    - Smoke-occluded objects focus primarily on the infrared modality; well-illuminated objects focus primarily on the visible modality.
    - Sampling points can adaptively align to the misaligned object locations.

### Loss & Training

- Follows the training loss of DETR-like detectors: $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{dn}$
- $\mathcal{L}_{cls}$: IoU-aware classification loss (RT-DETR)
- $\mathcal{L}_{box}$: L1 loss + GIoU loss
- $\mathcal{L}_{dn}$: Denoising training loss (DN-DETR)
- Auxiliary optimization loss is computed for each decoder layer.
- The backbone uses COCO pre-trained weights, ResNet50, trained for 20-50 epochs.
- Settings: $H=8$ attention heads, $K=4$ sampling points, $N=300$ queries.

## Key Experimental Results

### Main Results

**M3FD Dataset (most challenging, multi-scenario and multi-category)**:

| Method | Backbone | mAP50 | mAP75 | mAP |
|------|------|:---:|:---:|:---:|
| Yolov7 (RGB) | CSPDarknet53 | 69.0 | - | 42.7 |
| DINO (RGB) | ResNet50 | 73.3 | 48.2 | 46.3 |
| CFT | CSPDarknet53 | 68.2 | 44.6 | 42.5 |
| ICAFusion | CSPDarknet53 | 67.8 | 44.5 | 41.9 |
| **DAMSDet** | ResNet50 | **80.2** | **56.0** | **52.9** |

Compared to CFT, Ours achieves an improvement of 12.0% mAP50 and 10.4% mAP on M3FD.

**FLIR Dataset**:

| Method | mAP50 | mAP |
|------|:---:|:---:|
| ICAFusion | 79.2 | 41.4 |
| LRAF-Net | 80.5 | 42.8 |
| **DAMSDet** | **86.6** | **49.3** |

Improves by 6.1% mAP50 and 6.5% mAP compared to the previous best method.

### Ablation Study

Module ablation on M3FD:

| Configuration | mAP50 | mAP |
|------|:---:|:---:|
| Baseline (Bi-modal additive fusion + standard query) | 77.8 | 51.6 |
| + MCQS (Modality Competitive Selection) | 78.9 | 52.3 |
| + MDCA (Multispectral Deformable Cross-Attention) | 79.4 | 52.5 |
| + MCQS + MDCA + CQS | **80.2** | **52.9** |

- MCQS contributes +0.7% mAP: avoids introducing distracting modality information early on.
- MDCA contributes +0.9% mAP: enables fine-grained multi-level complementary information mining.
- CQS (Content Query Selection) provides a stronger prior in multi-modal scenarios.

### Key Findings

- **Significant Gap between Single-Modality and Multi-Modality**: Using only infrared on M3FD yields mAP=35.0, whereas incorporating visible fusion achieves 52.9.
- **Efficacy of the Cascaded Structure**: Clear differences in sampling positions and weight distributions across different decoder layers validate the necessity of layer-by-layer refinement of complementary information.
- **Modality Competitive Selection is Intuitive**: Visualization shows that infrared is selected under low-light conditions, while visible light is chosen under clear conditions.
- **Improves mAP50 by 5.6% on the small-object dataset VEDAI**, but the mAP is slightly lower than some CNN-based methods.

## Highlights & Insights

- **Simultaneously Solving Fusion and Alignment**: MDCA handles both complementary information fusion and modality misalignment unified within a single module, which is more efficient than processing them separately.
- **Dynamic Competition Replacing Global Fusion**: Selects the dominant modality independently for each object, avoiding an all-encompassing, one-size-fits-all fusion strategy.
- **Multi-Semantic Level Fusion**: The complementary characteristics of different semantic levels are also dynamically changing (infrared $\to$ low-level, visible $\to$ high-level).
- **No Extra Paired Annotations Required**: Unlike AR-CNN, it does not require dual-modality paired bounding box annotations to learn offsets.

## Limitations & Future Work

- Performance degrades during extreme misalignment where objects exceed the range of the 4D reference points.
- For small object detection, the bounding box regression accuracy of Transformers is inferior to that of CNNs, and the mAP metric is sometimes lower than that of LRAF-Net.
- The backbone adopts ResNet50; stronger backbones or larger DETR variants have not been explored.
- Performance could be further enhanced by incorporating infrared-visible image registration methods.

## Related Work & Insights

- **Deformable DETR**: The direct baseline for MDCA, extending deformable attention to dual modalities.
- **RT-DETR**: Borrows its Efficient Encoder and IoU-aware Query Selection.
- **DINO**: Borrows its cascaded structure and DN training strategy.
- **Insight**: The query mechanism of the DETR family is naturally suited for multi-modal scenarios, allowing competitive selection of the optimal modality.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[ECCV 2024\] Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)
- [\[CVPR 2026\] DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection](../../CVPR2026/object_detection/dyfclt_dynamic_frequency-decoupled_cross-modal_learning_transformer_for_multimod.md)
- [\[ECCV 2024\] DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)

</div>

<!-- RELATED:END -->
