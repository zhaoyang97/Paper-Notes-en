---
title: >-
  [Paper Note] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images
description: >-
  [CVPR2025][Object Detection][Visible-Infrared Fusion] This paper proposes ESM-YOLO+, a lightweight visible-infrared fusion network. By incorporating the MEAF module (pixel-level fusion with learnable spatial masks and spatial attention) and structural representation enhancement (SR, a super-resolution auxiliary supervision with zero inference overhead), it achieves 84.71% mAP on VEDAI with only 5.1M parameters (a 93.6% reduction).
tags:
  - "CVPR2025"
  - "Object Detection"
  - "Visible-Infrared Fusion"
  - "Small Object Detection"
  - "Spatial Mask"
  - "Lightweight Network"
  - "Training-time Super-Resolution"
date: 2026-05-08
content_hash: 08c768e0979be478
---

# Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images

**Conference**: CVPR2025  
**arXiv**: [2603.06925](https://arxiv.org/abs/2603.06925)  
**Code**: Pending confirmation  
**Area**: Object Detection  
**Keywords**: Visible-Infrared Fusion, Small Object Detection, Spatial Mask, Lightweight Network, Training-time Super-Resolution

## TL;DR
This paper proposes ESM-YOLO+, a lightweight visible-infrared fusion network. By incorporating the MEAF module (pixel-level fusion with learnable spatial masks and spatial attention) and structural representation enhancement (SR, a super-resolution auxiliary supervision with zero inference overhead), it achieves 84.71% mAP on VEDAI with only 5.1M parameters (a 93.6% reduction).

## Background & Motivation
**Task Scenario**: Small object detection in UAV/satellite remote sensing images, where targets occupy extremely low pixel proportions, exhibit weak textures, and are easily obscured by complex backgrounds.

**Multimodal Complementarity**: Visible images provide rich shape and texture features, while infrared images offer thermal radiation saliency; fusing them enhances small target representations.

**Limitations of Prior Work**:
   - Cross-modal feature heterogeneity (different scales, textures, and thermal characteristics) makes discriminative feature extraction difficult.
   - Spatial/temporal misalignment (sensor viewpoint, acquisition time difference, UAV motion) degrades fusion quality.
   - Existing fusion networks (e.g., MIR-YOLO, DVIF-Net) are structurally complex and unsuitable for resource-constrained edge deployment.

**Core Idea**: The fundamental challenge in visible-infrared small object detection lies not in aggressive feature enhancement, but in constructing modality-invariant and spatially consistent representations—requiring a reliable conditional fusion in a "mask-then-attention" manner.

## Method

### ESM-YOLO+ Overall Architecture
Three core components: MEAF pixel-level fusion module $\rightarrow$ Detection Backbone + Detection Head $\rightarrow$ Training-time SR super-resolution auxiliary branch (removed during inference).

### MEAF (Mask-Enhanced Attention Fusion)

1. **Modality Scaling**: Learnable parameters $p^{RGB}, p^{IR}$ weight the two modality inputs.
2. **Spatial Mask Generation**: Uses 3×3 Conv + ReLU + 1×1 Conv to generate spatial masks for each modality, element-wise multiplying them by original features to highlight salient structures.
3. **Residual Connection + Convolutional Aggregation**: Masked features + original features $\rightarrow$ 3×3 Conv refinement.
4. **Spatial Attention**: Average pooling and max pooling along the channel dimension $\rightarrow$ concatenation $\rightarrow$ 1×1 Conv + Sigmoid to generate spatial attention maps.
5. **Modality Fusion**: Attention-modulated dual-branch features are concatenated $\rightarrow$ weighted by a channel excitation vector $M$ $\rightarrow$ outputting fused features.

### SR Training-time Structural Representation Enhancement
- Extracts features from intermediate backbone layers $\rightarrow$ reconstructs images via a lightweight decoder (upsampling + convolution).
- Loss $\mathcal{L}_{SR} = \|S - \mathcal{D}(X)\|_1$, forcing the backbone features to preserve fine-grained spatial structures.
- **Gradients are backpropagated to the backbone during training**, and the SR auxiliary branch is discarded during inference $\rightarrow$ zero inference overhead.

### Total Loss
$\mathcal{L}_{total} = c_1 \mathcal{L}_{\text{det}} + c_2 \mathcal{L}_{SR}$

## Key Experimental Results

### SOTA Comparison on VEDAI Dataset

| Method | mAP50 | Parameters |
|------|-------|--------|
| YOLOv5x | 62.65 | — |
| SuperYOLO | 75.09 | — |
| ESM-YOLO (baseline) | 82.42 | — |
| ACDF-YOLO | 78.1 | — |
| CFT (Transformer) | 85.3 | 196.9M |
| **ESM-YOLO+** | **84.71** | **5.1M** |

- mAP increases by +2.29% compared to the baseline ESM-YOLO.
- Parameter size is only 5.1M, a 93.6% reduction compared to baseline, and GFLOPs are reduced by 68.0%.
- Close in accuracy to the Transformer-based method CFT (85.3%), but with a 97.4% reduction in parameters.

### DroneVehicle Dataset
- Achieves 74.0% mAP, validating effectiveness on a larger-scale dataset.

### Ablation on Single Modality vs Multimodality (VEDAI)

| Method | mAP50 |
|------|-------|
| Single Modality RGB | 75.06 |
| Single Modality IR | 70.83 |
| **Multimodal ESM-YOLO+** | **84.71** |

- Multimodal fusion yields a +9.65% improvement.

### Comparison with Dedicated Multimodal Small Object Detection Methods (VEDAI)

| Method | mAP50 |
|------|-------|
| MIR-YOLO | 69.7 |
| ACDF-YOLO | 78.1 |
| MMFDet | 77.9 |
| FFCA-YOLO | 74.8 |
| **ESM-YOLO+** | **84.7** |

- Improves mAP by 6.6% compared to the second-best ACDF-YOLO.
- Achieves optimal results in 6 out of 8 categories.

### Training Details
- SGD optimizer, momentum=0.937, weight decay=0.0005, 300 epochs, batch=2.
- Loss weights: $\alpha_o^a=1.0, \alpha_l^a=0.05, \alpha_c^a=0.5$.
- Modality fusion parameters $p^{RGB} = p^{IR} = 0.5$.
- Trained at 1024×1024 resolution and tested at 512×512.

## Highlights & Insights
- **Two-Step Mask-Attention Fusion Paradigm**: First utilizes a learnable spatial mask for alignment-aware selection, then employs spatial attention to maintain the spatial support of small targets, fundamentally addressing cross-modal heterogeneity.
- **Zero Inference Cost of Training-time SR**: Cleverly leverages an auxiliary task to regularize the backbone, which is directly discarded during inference.
- **Extreme Lightweighting**: Achieves accuracy comparable to a 196.9M-parameter Transformer method with only 5.1M parameters.
- **Practical Deployment Friendly**: CNN architecture with low computational complexity, suitable for edge devices and real-time scenarios.

## Limitations & Future Work
- The scale of the VEDAI dataset is relatively small (1246 image pairs); generalization needs further verification on larger datasets.
- MEAF fusion assumes that RGB and IR are basically aligned; severely misaligned scenarios may require an additional registration module.
- The sensitivity of the SR auxiliary branch hyperparameter $c_2$ to performance is not fully discussed.
- Supports only dual-modality (RGB+IR); extending to more modalities requires architectural adjustments.
- Comparisons with the latest state-of-the-art dedicated multimodal small object detection methods could be further expanded.

## Rating
- Novelty: ⭐⭐⭐ MEAF is a reasonable enhancement of BEF, and the SR training technique is novel but not conceptually groundbreaking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive with two datasets, multi-method comparisons, modality ablation, and fusion visualizations.
- Writing Quality: ⭐⭐⭐ Well-structured, but some definitions are somewhat verbose.
- Value: ⭐⭐⭐⭐ The lightweight and efficient practical solution holds clear value for edge deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Evaluation of Infrared Small Target Detection](../../NeurIPS2025/object_detection/rethinking_evaluation_of_infrared_small_target_detection.md)
- [\[ICCV 2025\] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)
- [\[CVPR 2026\] Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection](../../CVPR2026/object_detection/target-aware_invertible_encoder_with_reconstruction_guidance_for_infrared_small_.md)
- [\[ICCV 2025\] From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](../../ICCV2025/object_detection/from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)
- [\[CVPR 2026\] Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images](../../CVPR2026/object_detection/balanced_hierarchical_contrastive_learning_with_decoupled_queries_for_fine-grain.md)

</div>

<!-- RELATED:END -->
