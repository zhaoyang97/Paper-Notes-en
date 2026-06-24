---
title: >-
  [Paper Note] DEIM: DETR with Improved Matching for Fast Convergence
description: >-
  [CVPR 2025][Object Detection][DETR Acceleration] This paper accelerates DETR training convergence through two simple improvements: Dense O2O (increasing targets per image via data augmentation to achieve dense one-to-one matching) and MAL (replacing VFL to better optimize low-quality matches). It cuts the training epochs in half while boosting performance (COCO AP 56.5 with D-FINE-X).
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "DETR Acceleration"
  - "One-to-One Matching Improvement"
  - "Dense Supervision"
  - "Matchability-Aware Loss"
date: 2026-05-08
content_hash: 61e9da1caac48f81
---

# DEIM: DETR with Improved Matching for Fast Convergence

**Conference**: CVPR 2025  
**arXiv**: [2412.04234](https://arxiv.org/abs/2412.04234)  
**Code**: [https://www.shihuahuang.cn/DEIM/](https://www.shihuahuang.cn/DEIM/)  
**Area**: Object Detection  
**Keywords**: DETR Acceleration, One-to-One Matching Improvement, Dense Supervision, Matchability-Aware Loss, Object Detection

## TL;DR
This paper accelerates DETR training convergence through two simple improvements: Dense O2O (increasing targets per image via data augmentation to achieve dense one-to-one matching) and MAL (replacing VFL to better optimize low-quality matches). It cuts the training epochs in half while boosting performance (COCO AP 56.5 with D-FINE-X).

## Background & Motivation

**Background**: The DETR series employs Hungarian matching for one-to-one (O2O) label assignment, which converges more slowly than YOLO's one-to-many (O2M) matching. O2M provides denser training signals but requires NMS post-processing.

**Limitations of Prior Work**: (1) O2O matching assigns only one positive sample per object, resulting in sparse training signals. (2) Low-quality matches (low IoU) receive near-zero gradients under VFL, leaving them severely under-optimized. (3) Adding auxiliary O2M decoders (e.g., Group DETR) provides more positive samples but increases model complexity.

**Key Challenge**: O2O matching ensures end-to-end NMS-free inference but suffers from sparse training signals. The challenge lies in increasing the density of training signals without introducing additional decoders.

**Goal**: To improve the training efficiency of O2O matching without introducing extra model complexity, while enhancing the loss function's capability to handle low-quality matches.

**Key Insight**: Leveraging Mosaic/MixUp data augmentation to concatenate multiple images into one—increasing the number of targets per image from ~6 to ~24, which naturally scales up the number of positive samples in O2O matching by 4×. Concurrently, MAL replaces VFL to assign larger gradients to low-IoU matches.

**Core Idea**: Utilizing data augmentation to increase target density per image to achieve "dense O2O without extra decoders" + replacing VFL with MAL to improve the optimization of low-quality matches.

## Method

### Overall Architecture
Mosaic/MixUp image concatenation ($4 \to 1$) $\rightarrow$ increase target count $N$ per image $\rightarrow$ maintain one-to-one matching (each target still matches only one query) $\rightarrow$ MAL loss replaces VFL to better optimize low-IoU positive samples $\rightarrow$ training scheduler enables Dense O2O in the first 4 epochs and turns off augmentation in the later stage.

### Key Designs

1. **Dense O2O**:

    - Function: Provides dense training signals without increasing model complexity.
    - Mechanism: Mosaic concatenates 4 images into one, increasing the targets per image from ~6 to ~24. Each target still matches only one query (O2O), but the total count of positive samples increases by 4×. This is effectively equivalent to the supervision density of O2M without requiring auxiliary decoders.
    - Design Motivation: Analysis shows SimOTA (O2M) generates ~10 positive samples per target on average. Dense O2O yields a comparable number of positive samples to SimOTA, making it "cost-free" in terms of computational overhead.

2. **Matchability-Aware Loss (MAL)**:

    - Function: Improves the optimization of low-quality matches (low IoU).
    - Mechanism: For positive sample loss $-q^\gamma \log(p) - (1-q^\gamma)\log(1-p)$, $q^\gamma$ ($\gamma=2$) replaces $q$ in VFL as the target. When IoU is very low (e.g., $q = 0.05$), the loss in VFL is close to zero, leading to vanishing gradients. In MAL, although $q^2 = 0.0025$ is small, the loss curve is steeper and the gradient does not vanish.
    - Design Motivation: Low-quality matches are highly common in early training stages (before the model learns well). If these matches are not optimized, model improvement is slow.

3. **Training Schedule**:

    - Function: Balances augmentation strength and learning stability.
    - Mechanism: Dense O2O performs warmup in the first 4 epochs, and data augmentation is turned off after 50% of the training (allowing the model to adapt to the real distribution without augmentation).
    - Design Motivation: Excessive augmentation can cause distribution shift. Disabling it at the right time allows the model to converge on the real data distribution.

### Loss & Training
MAL is used for classification, and GIoU + L1 for regression. It is compatible with RT-DETR and D-FINE architectures. Training can be completed on a single NVIDIA 4090 GPU.

## Key Experimental Results

### Main Results

| Model | Epoch | AP | AP50 | Latency |
|------|-------|-----|------|------|
| YOLOv10-X | 500 | 54.4 | 71.3 | 10.74ms |
| YOLO11-X | 500 | 54.1 | 70.8 | 10.52ms |
| D-FINE-L | 72 | 54.0 | 71.6 | 8.07ms |
| **DEIM-D-FINE-L** | **50** | **54.7** | **72.4** | **8.07ms** |
| D-FINE-X | 72 | 55.8 | 73.7 | 12.89ms |
| **DEIM-D-FINE-X** | **50** | **56.5** | **74.0** | **12.89ms** |

### Ablation Study

| Component | AP $\Delta$ | Description |
|------|------|------|
| Baseline D-FINE-L | 54.0 | 72 epochs |
| +Dense O2O | +0.4 | Dense positive samples |
| +MAL | +0.3 | Low IoU optimization improvement |
| +Both 50ep | **54.7** | **30% reduction in epochs with performance gain** |

### Key Findings
- **Performance surpasses with halved training epochs**: DEIM-D-FINE with 50 epochs outperforms D-FINE with 72 epochs and YOLOv10/11 with 500 epochs.
- **Dense O2O is a free lunch**: It improves training efficiency solely through data augmentation without adding model parameters or inference latency.
- **Steep gradient of MAL for low-quality matches**: When IoU=0.05, the gradient of VFL is near-zero, while MAL still provides effective gradients.

## Highlights & Insights
- **The philosophy of "using data augmentation to replace additional decoders" is exceptionally elegant**: It achieves O2M-level supervision density while preserving the end-to-end advantages of O2O.
- **In-depth design analysis of MAL**: The gradient curve comparison clearly demonstrates the flaws of VFL under low IoU.
- **Trainable on a single 4090 GPU**: This means top-tier detectors can be easily reproduced even in academic lab settings.

## Limitations & Future Work
- Dense O2O relies on Mosaic augmentation, which might not be suitable for certain datasets.
- The $\gamma = 2$ in MAL is an empirical value, which might need tuning on different datasets.
- Validated only on the COCO dataset.

## Related Work & Insights
- **vs Group DETR / DN-DETR**: These methods use extra decoders or denoising queries to provide more positive samples. DEIM requires no extra modules.
- **vs RT-DETRv2 / D-FINE**: DEIM further improves AP by 0.5-0.7 on these backbones while reducing training time.

## Rating
- Novelty: ⭐⭐⭐⭐ Dense O2O and MAL are simple individually but yield notable performance when combined.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across multiple backbones, epochs, compared with the YOLO series, and featuring detailed gradient analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear matching analysis.
- Value: ⭐⭐⭐⭐⭐ Direct engineering value for improving DETR training efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](../../ECCV2024/object_detection/tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)
- [\[ICCV 2025\] Sim-DETR: Unlock DETR for Temporal Sentence Grounding](../../ICCV2025/object_detection/sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [\[NeurIPS 2025\] ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction](../../NeurIPS2025/object_detection/recon-gs_continuum-preserved_gaussian_streaming_for_fast_and_compact_reconstruct.md)

</div>

<!-- RELATED:END -->
