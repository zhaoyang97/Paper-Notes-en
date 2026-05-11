---
title: >-
  [Paper Note] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation
description: >-
  [CVPR 2026][Segmentation][Open-vocabulary segmentation] This paper proposes SPAR, which distills the spatial reasoning capability of a fine-stride sliding window teacher into a single-pass student of identical architecture, transforming a ViT into a resolution-agnostic dense feature extractor. SPAR achieves +10.5 mIoU over the single-pass baseline in open-vocabulary segmentation while running 52× faster than the teacher.
tags:
  - CVPR 2026
  - Segmentation
  - Open-vocabulary segmentation
  - resolution-agnostic
  - knowledge distillation
  - Vision Transformer
  - sliding window inference
date: 2026-05-08
content_hash: 277ed582dfb6f2bd
---

# SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation

**Conference**: CVPR 2026
**arXiv**: [2604.02252](https://arxiv.org/abs/2604.02252)
**Code**: [https://github.com/naomikombol/SPAR](https://github.com/naomikombol/SPAR)
**Area**: Segmentation / Open-Vocabulary Segmentation
**Keywords**: Open-vocabulary segmentation, resolution-agnostic, knowledge distillation, Vision Transformer, sliding window inference

## TL;DR

This paper proposes SPAR, which distills the spatial reasoning capability of a fine-stride sliding window teacher into a single-pass student of identical architecture, transforming a ViT into a resolution-agnostic dense feature extractor. SPAR achieves +10.5 mIoU over the single-pass baseline in open-vocabulary segmentation while running 52× faster than the teacher.

## Background & Motivation

**Background**: Foundation ViTs (CLIP, SigLIP2, DINOv3) excel at image-level understanding via contrastive/self-supervised learning, but are limited in dense prediction tasks (e.g., segmentation) that require fine-grained spatial understanding, due to fixed-resolution pretraining and coarse patch-level representations. Open-vocabulary segmentation (OVS) requires models to segment arbitrary categories from text alone, placing even higher demands on fine-grained pixel-level reasoning at high resolution.

**Limitations of Prior Work**: Two strategies exist for handling high-resolution images: (1) single-pass inference with interpolated positional encodings — efficient but inaccurate, as the train-inference resolution mismatch distorts positional information; (2) sliding window inference — significantly improves accuracy via small-stride overlapping windows (each patch appears in multiple contexts), but at extremely high computational cost. For example, sliding window with stride 24 is ~52× slower than single-pass inference.

**Key Challenge**: There is a severe accuracy–efficiency trade-off — single-pass inference is fast but weak; sliding window inference is strong but slow. Existing resolution adaptation methods (e.g., NaFlex) work well for image-level tasks but underperform on dense prediction.

**Goal**: Achieve segmentation accuracy comparable to or exceeding fine-stride sliding window inference while retaining the efficiency of single-pass forward inference.

**Key Insight**: The advantage of sliding window inference fundamentally stems from sub-patch regions being exposed to diverse contexts and robustness gained through averaging — a form of spatial reasoning that can be transferred to a single-pass model via distillation.

**Core Idea**: Distill the spatial features of a fine-stride sliding window teacher into a single-pass student of identical architecture using a feature regression loss, requiring no architectural modification or pixel-level annotation.

## Method

### Overall Architecture

**Teacher**: A frozen VLM visual encoder operating in sliding window mode (window size = pretraining resolution, stride = 24), extracting features per window and stitching them into a unified feature map. **Student**: Initialized from the same architecture, performing single-pass forward inference on the full high-resolution image to produce a feature map. The training objective is to make the student's feature map approximate the teacher's stitched feature map. At inference, only the student is used, enabling efficient and accurate segmentation.

### Key Designs

1. **Sliding Window Teacher Feature Stitching**:

    - **Function**: Generate high-quality dense features as distillation targets.
    - **Mechanism**: A high-resolution image $X \in \mathbb{R}^{3 \times H \times W}$ is divided into $m$ overlapping windows of size $K \times K$ with stride $s=24$ (non-divisible by patch size $P=16$). Each window is independently encoded, then upsampled (factor $r=2$) and averaged to form the stitched feature map: $V_\text{teacher}(X) = \text{stitch}(\{f(X_{w_i})\}_{i=1}^m)$. The non-divisible stride exposes sub-patch regions to more diverse contexts, further improving quality.
    - **Design Motivation**: Small stride and high overlap ensure each pixel is observed in multiple windows, yielding robustness analogous to test-time augmentation through averaging.

2. **Feature Distillation Training**:

    - **Function**: Transfer the teacher's spatial reasoning capability to the single-pass student.
    - **Mechanism**: The student $g$ performs single-pass inference on the full image to produce $V_\text{student}(X) = g(X)$. The distillation loss is a simple MSE: $\mathcal{L}_\text{distill} = \|V_\text{teacher}(X) - V_\text{student}(X)\|_2^2$. Training uses diverse resolutions and aspect ratios (short side 512–2048 pixels). A key finding is that unfreezing only the last 2 blocks suffices for strong performance under standard OVS settings, while full fine-tuning is preferable for very high-resolution inference.
    - **Design Motivation**: MSE loss is direct and efficient, requiring no pixel-level annotations. Teacher features are precomputed and stored for reuse; training requires only 25k SA-1B images and approximately 1.5 hours on 2×A6000 GPUs.

3. **Resolution and Aspect Ratio Augmentation**:

    - **Function**: Make the student robust to diverse resolutions and aspect ratios.
    - **Mechanism**: During training, images are randomly rescaled (short side 512–2048), randomly cropped (side length from 512 to maximum possible), and horizontally flipped. All images are bilinearly resampled to dimensions divisible by the patch size. An extended variant further expands the short-side range to 512–2560 and trains all parameters to support higher-resolution inference.
    - **Design Motivation**: Exposure to diverse resolutions promotes generalization of positional encodings, outperforming NaFlex which is pretrained at a single fixed resolution.

### Loss & Training

Pure feature regression loss (MSE) with no annotations required. AdamW optimizer, constant learning rate $2 \times 10^{-5}$, weight decay $10^{-4}$, trained for 10 epochs. By default, only the last 2 blocks are fine-tuned. Teacher features are precomputed and stored (~170 GB) to avoid redundant computation. Batch size = 1 (due to variable-length sequences).

## Key Experimental Results

### Main Results

Average mIoU across 6 datasets for SigLIP2 – ViT-B-16:

| Method | Inference Mode | Mean₆ |
|--------|----------------|-------|
| NaFlex | Single-pass | 31.7 |
| Pre-trained | Single-pass | 33.1 |
| Pre-trained | Sliding window (s=24) | 41.2 |
| **SPAR** | **Single-pass** | **43.6** |
| SPAR + AnyUp | Single-pass | 46.8 |
| SPAR + LPOSS | Single-pass | 46.7 |

SPAR surpasses the single-pass baseline by +10.5 mIoU and even outperforms the teacher (sliding window, s=24) by +2.4 mIoU.

### Ablation Study

Improvement across different backbones:

| Backbone | Single-pass Baseline | SPAR | Gain |
|----------|----------------------|------|------|
| SigLIP2 ViT-B-16 | 33.1 | 43.6 | +10.5 |
| OpenCLIP ViT-B-16 | 27.7 | 34.4 | +6.7 |
| DINOv3 ViT-L-16 | 43.8 | 44.4 | +0.6 |

### Key Findings

- **NaFlex is ill-suited for dense prediction**: Despite SigLIP2's dedicated NaFlex design for resolution adaptation, it underperforms both SPAR and standard sliding window inference on OVS, demonstrating that image-level resolution adaptation does not equate to patch-level spatial understanding.
- **Non-divisible stride outperforms divisible stride**: $s=24$ outperforms $s=32$ (divisible by 16), as sub-patch regions are exposed to more diverse contexts.
- **Student surpasses teacher**: SPAR exceeds the teacher on average mIoU and on most individual datasets, likely due to implicit regularization from multi-resolution training during distillation.
- **Modest gains on DINOv3**: DINOv3 already incorporates RoPE encodings and high-resolution fine-tuning, making its single-pass baseline relatively strong. Nevertheless, SPAR still improves Cityscapes mIoU from 35.9 to 40.1 (a high-resolution benchmark).
- SPAR is complementary to methods such as AnyUp and LPOSS, with further gains when combined.

## Highlights & Insights

- **Extreme simplicity**: No architectural modifications, no pixel-level annotations, no complex loss functions — a remarkable improvement achieved solely through MSE feature distillation.
- **52× speedup**: Maintaining single-pass inference efficiency compared to sliding window inference with stride 24 represents substantial practical value.
- **Strong generality**: Effective across three architecturally distinct backbones — SigLIP2, OpenCLIP, and DINOv3.
- **Minimal training cost**: Only 25k unannotated images and ~1.5 hours of training are required.
- The work reveals an important insight: the advantage of sliding window inference can be distilled, and the resulting student can even surpass the original teacher.

## Limitations & Future Work

- Models already robust to resolution variation (e.g., DINOv3) leave limited room for improvement.
- Storing teacher features requires ~170 GB, which may pose storage constraints.
- Validation is limited to the training-free OVS setting; performance on supervised methods or other dense prediction tasks (detection, depth estimation) remains unexplored.
- Batch size is constrained to 1 due to variable-length sequences, leaving room for training efficiency improvements.
- Higher-order distillation strategies (e.g., attention distillation) beyond pure feature regression are worth exploring.

## Related Work & Insights

- **FlexiViT / NaViT / ResFormer**: Enhance resolution robustness through multi-resolution pretraining, but require training from scratch.
- **SigLIP2 NaFlex**: Unifies flexible patching and variable-length sequences, but this paper demonstrates its insufficiency for dense prediction.
- **LPOSS**: A training-free label propagation method complementary to SPAR (+3.1 mIoU when combined).
- The distillation paradigm is extensible to other scenarios requiring high-resolution dense inference, such as video understanding and 3D vision.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Transferring the benefit of sliding window inference to single-pass inference via distillation is an insightful contribution, though distillation itself is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 3 backbones × 6 datasets × multiple resolutions × combinations with various methods.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, intuitive trade-off figures, and concise method description.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical — simple, general, efficient, and effective; broadly applicable to any scenario requiring high-resolution ViT inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation](pca-seg_revisiting_cost_aggregation_for_openvocabulary_semantic_and_part_segmentat.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)

</div>

<!-- RELATED:END -->
