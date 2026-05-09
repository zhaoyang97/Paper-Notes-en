---
title: >-
  [Paper Note] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images
description: >-
  [CVPR 2026][Object Detection][Small target detection] This paper proposes ESM-YOLO+, a lightweight visible-infrared fusion network for small target detection. It achieves pixel-level cross-modal adaptive fusion via a Mask-Enhanced Attention Fusion (MEAF) module, and introduces a training-time structural representation enhancement to improve spatial discriminability. The method achieves 84.71% mAP on VEDAI while reducing parameter count by 93.6%.
tags:
  - CVPR 2026
  - Object Detection
  - Small target detection
  - visible-infrared fusion
  - remote sensing
  - spatial attention
  - lightweight network
date: 2026-05-08
content_hash: 623a7e56901330a0
---

# Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images

**Conference**: CVPR 2026
**arXiv**: [2603.06925](https://arxiv.org/abs/2603.06925)
**Code**: None
**Area**: Object Detection
**Keywords**: Small target detection, visible-infrared fusion, remote sensing, spatial attention, lightweight network

## TL;DR

This paper proposes ESM-YOLO+, a lightweight visible-infrared fusion network for small target detection. It achieves pixel-level cross-modal adaptive fusion via a Mask-Enhanced Attention Fusion (MEAF) module, and introduces a training-time structural representation enhancement to improve spatial discriminability. The method achieves 84.71% mAP on VEDAI while reducing parameter count by 93.6%.

## Background & Motivation

**Background**: Small target detection in UAV remote sensing imagery is highly challenging due to limited target pixels and complex backgrounds. Visible-light images provide texture and structural information but are sensitive to illumination, while infrared images capture thermal radiation but lack fine-grained features. Multimodal fusion is an effective approach to improving detection performance.

**Limitations of Prior Work**: Existing multimodal detectors (e.g., MIR-YOLO, DVIF-Net) over-rely on complex fusion designs—incorporating heavyweight structures such as Vision-LSTM and gated aggregation—making them unsuitable for resource-constrained UAV and satellite platforms. More critically, they either assume perfect modal alignment or rely on dense feature encoding, without directly addressing cross-modal structural inconsistency.

**Key Challenge**: Direct fusion is essentially unconditional aggregation, which may amplify modality-dominant background responses or suppress spatially localized weak targets. Effective fusion should encode spatial reliability priors and condition cross-modal interaction on trustworthy regions.

**Goal**: (1) How to achieve lightweight yet effective pixel-level cross-modal fusion? (2) How to enhance spatial structural representation without incurring additional inference cost?

**Key Insight**: The paper proposes a "mask-then-attend" paradigm—learnable spatial masks perform alignment-aware selection, while spatial attention enforces topological consistency re-weighting.

**Core Idea**: Learnable spatial masks suppress unreliable cross-modal interactions; spatial attention preserves the spatial support of small targets; training-time super-resolution guidance enhances feature-space discriminability.

## Method

### Overall Architecture

ESM-YOLO+ consists of three components: (1) a MEAF module for pixel-level visible-infrared fusion; (2) a detection backbone and head based on the YOLO architecture; and (3) an EDSR super-resolution branch that provides auxiliary supervision during training only and is discarded at inference.

### Key Designs

1. **Mask-Enhanced Attention Fusion (MEAF)**:

    - **Function**: Pixel-level cross-modal fusion that adaptively highlights target-relevant regions.
    - **Mechanism**: For each modality—RGB ($I^{RGB}$) and IR ($I^{IR}$)—the following steps are applied: ① learnable modality scaling $I_1 = I \cdot p$; ② spatial mask generation $I_{mask} = I_1 \otimes \text{Conv}_{1\times1}(\sigma_R(\text{Conv}_{3\times3}(I_1)))$ to highlight salient structures; ③ convolutional aggregation with residual connection; ④ spatial attention $I_{SA} = \sigma_S(\text{Conv}(\text{Cat}(\bar{I}, \hat{I})))$, where $\bar{I}$ and $\hat{I}$ denote channel-wise average and max pooling; ⑤ a channel excitation vector $M$ globally re-weights the concatenated features.
    - **Design Motivation**: The spatial mask performs alignment-aware soft selection to suppress unreliable regions, while the spatial attention preserves the spatial topological support of small targets. The two mechanisms are complementary, jointly addressing cross-modal heterogeneity and scale alignment issues.

2. **Training-Time Structural Representation Enhancement (SR)**:

    - **Function**: Provides auxiliary gradient signals during training to enhance the backbone's spatial discriminability.
    - **Mechanism**: A lightweight decoder is attached to an intermediate backbone stage to perform image reconstruction, with an L1 loss $\mathcal{L}_{SR} = \|S - \mathcal{D}(X)\|_1$ aligning the reconstructed output with the downsampled input. The reconstruction branch is discarded after training.
    - **Design Motivation**: Without adding inference overhead, this branch forces the backbone features to retain fine-grained spatial topological information that is easily lost during aggressive downsampling.

### Loss & Training

Total loss: $\mathcal{L}_{total} = c_1 \mathcal{L}_{det} + c_2 \mathcal{L}_{SR}$. The detection loss includes objectness, localization, and classification terms. Optimizer: SGD with Nesterov momentum (lr = 0.01, momentum = 0.937), trained for 300 epochs on a single RTX 4060. Training resolution: 1024×1024; test resolution: 512×512.

## Key Experimental Results

### Main Results (VEDAI Dataset, mAP50 %)

| Method | Car | Truck | Boat | Van | mAP50 |
|--------|-----|-------|------|-----|-------|
| YOLOv5s | 80.81 | 54.71 | 24.25 | 45.96 | 56.79 |
| SuperYOLO | 91.13 | 70.18 | 60.24 | 76.50 | 75.09 |
| ESM-YOLO | 90.80 | 83.83 | 85.23 | 80.11 | 82.42 |
| **ESM-YOLO+** | **93.64** | **86.65** | **71.57** | **85.20** | **84.71** |

### Efficiency Comparison

| Metric | Baseline | ESM-YOLO+ | Reduction |
|--------|----------|-----------|-----------|
| Parameters | — | Extremely lightweight | **−93.6%** |
| GFLOPs | — | Extremely low | **−68.0%** |

### Key Findings

- ESM-YOLO+ improves mAP by 2.29% over the baseline ESM-YOLO while substantially reducing model complexity.
- The method achieves 74.0% mAP on DroneVehicle, validating its effectiveness in large-scale, multi-category scenarios.
- F1 curves during training are higher with the SR branch, confirming that auxiliary reconstruction gradients effectively enhance feature discriminability.
- Qualitative visualizations show that ESM-YOLO+ significantly reduces both false positives and missed detections.

## Highlights & Insights

- **Mask-then-attend paradigm**: Transforms fusion from unconditional aggregation to conditioned interaction—a conceptually clean and lightweight approach transferable to other multimodal fusion tasks.
- **Training-time enhancement at zero inference cost**: The SR branch serves only as a gradient-level regularizer during training and is absent at inference, making this strategy particularly valuable for lightweight models.
- **93.6% parameter reduction**: The model improves accuracy while drastically reducing complexity, demonstrating the efficiency of the proposed design.

## Limitations & Future Work

- MEAF assumes that RGB and IR images are coarsely aligned; severe spatial misalignment may require additional geometric correction.
- Evaluation is limited to vehicular small targets; generalization to other categories (pedestrians, buildings, etc.) warrants further investigation.
- The 4-factor spatial partition is fixed as a uniform division; an adaptive, semantics-based partitioning strategy may yield better results.

## Related Work & Insights

- **vs. ESM-YOLO**: ESM-YOLO employs BEF (simple convolutional excitation), whereas ESM-YOLO+ uses MEAF (mask + spatial attention), achieving a 2.29% mAP gain.
- **vs. SuperYOLO**: SuperYOLO achieves acceptable accuracy but at high computational cost; ESM-YOLO+ surpasses it in accuracy with significantly lower complexity.

## Rating

- Novelty: ⭐⭐⭐ — An incremental improvement over existing methods, though the mask-then-attend paradigm carries some originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, comprehensive comparisons, and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and well-formulated equations.
- Value: ⭐⭐⭐⭐ — Practically relevant for lightweight small target detection in remote sensing.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[ICCV 2025\] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)
- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)
- [\[CVPR 2026\] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding](mrd_multi-resolution_retrieval-detection_fusion_for_high-resolution_image_unders.md)

<!-- RELATED:END -->
