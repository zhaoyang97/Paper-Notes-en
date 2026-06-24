---
title: >-
  [Paper Note] Segment Anything, Even Occluded
description: >-
  [CVPR 2025][Autonomous Driving][Occlusion Segmentation] Proposes SAMEO, which adapts EfficientSAM as an amodal segmentation decoder for occluded objects. Combined with a newly constructed 300K-image Amodal-LVIS dataset, it achieves zero-shot amodal segmentation performance on COCOA-cls and D2SA that outperforms supervised methods.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Occlusion Segmentation"
  - "SAM Adaptation"
  - "Synthetic Dataset"
  - "Zero-shot Generalization"
  - "Foundation Models"
date: 2026-05-08
content_hash: 2bb43a6cb3a99391
---

# Segment Anything, Even Occluded

**Conference**: CVPR 2025  
**arXiv**: [2503.06261](https://arxiv.org/abs/2503.06261)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: Occlusion Segmentation, SAM Adaptation, Synthetic Dataset, Zero-shot Generalization, Foundation Models

## TL;DR

Proposes SAMEO, which adapts EfficientSAM as an amodal segmentation decoder for occluded objects. Combined with a newly constructed 300K-image Amodal-LVIS dataset, it achieves zero-shot amodal segmentation performance on COCOA-cls and D2SA that outperforms supervised methods.

## Background & Motivation

Amodal instance segmentation aims to predict the complete shape of objects (including occluded parts) and has important applications in autonomous driving, robotic manipulation, and scene understanding. Existing methods suffer from the following limitations:

1. **Lack of Flexibility**: Existing methods require joint training of front-end detectors and mask decoders, making them unable to leverage existing powerful pretrained detectors.
2. **Limited Dataset Scale**: Existing amodal datasets have a small number of images, and the annotation quality is highly inconsistent.
3. **Annotation Bias**: Many datasets contain numerous meaningless annotations (e.g., walls, floors), contributing minimally to scene understanding.
4. **Synthetic Data Issues**: Automatically generated datasets suffer from inconsistent and erroneous instance annotations.

Foundation models such as SAM perform exceptionally well on modal segmentation but cannot directly handle occluded regions. The core idea of this work is to extend SAM's capabilities to amodal segmentation while preserving its zero-shot generalization capacity.

## Method

### Overall Architecture

SAMEO is based on the EfficientSAM architecture, retaining the lightweight image encoder $\mathcal{E}$, the prompt encoder $\mathcal{P}$, and the two-way cross-attention mask decoder $\mathcal{D}$. Given an image $I$ and a box prompt $B$, it predicts the amodal mask $\hat{M}$ and the estimated IoU $\hat{\rho}$:

$$\hat{M}, \hat{\rho} = \mathcal{D}(\mathcal{E}(I), \mathcal{P}(B))$$

During inference, various front-end detectors (such as AISFormer or RTMDet) can be flexibly integrated, where the bounding boxes serve as prompts input into SAMEO to generate the amodal masks.

### Key Designs

**1. Fine-tuning Mask Decoder Only Training Strategy**

- **Function**: Fine-tune only the mask decoder to adapt to amodal segmentation while keeping the weights of the image encoder and prompt encoder fixed.
- **Core Idea**: During training, randomly select either modal or amodal ground-truth boxes as prompts with equal probability, enabling the model to learn amodal prediction capabilities under both types of prompts concurrently.
- **Design Motivation**: To maintain the pretrained representation capacity of the encoders and avoid overfitting on limited amodal data, while improving compatibility with different front-end detectors via the random prompting strategy.

**2. Amodal-LVIS Large-scale Synthetic Dataset**

- **Function**: Provides paired training data of 300K images, where each image contains instance annotations of both occluded and unoccluded versions.
- **Core Idea**: Collects fully unoccluded objects from LVIS/LVVIS, randomly pairs them to generate synthetic occlusions, and employs a dual-labeling mechanism (retaining both occluded and original versions).
- **Design Motivation**: Training exclusively on occluded instances leads the model to over-predict the background as occluded objects (over-prediction bias); the dual-labeling mechanism effectively prevents this occlusion bias.

**3. Comprehensive Dataset Cleaning and Collection**

- **Function**: Constructs a comprehensive training set comprising 1M images and 2M instance annotations.
- **Core Idea**: Filters meaningless architectural elements from DYCE and MP3D-amodal, applies occupancy/occlusion thresholds to filter out unnatural occlusions in WALT, and filters "stuff" annotations from COCOA, etc.
- **Design Motivation**: Existing datasets suffer from annotation noise and irrelevant objects; systematic cleaning ensures the quality of the training data.

### Loss & Training

The training loss combines Dice loss, Focal loss, and an L1 loss for IoU estimation:

$$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{Focal}} + \lambda \mathcal{L}_{\text{IoU}}$$

where $\lambda = 0.05$ and $\gamma = 2$ in the Focal loss. The IoU prediction is used during inference to refine the confidence from the front-end detector: $\hat{\rho}_{\text{ref}} = \hat{\rho}_{\text{front}} \times \hat{\rho}_{\text{ours}}$.

## Key Experimental Results

### Main Results: Performance Comparison under Different Front-end Detectors (COCOA-cls / D2SA)

| Method | COCOA-cls AP | COCOA-cls AP50 | D2SA AP | D2SA AP50 |
|------|-------------|---------------|---------|----------|
| AISFormer | 40.6 | 70.5 | 66.3 | 89.9 |
| RTMDet* | 49.8 | 71.2 | 59.7 | 81.3 |
| AISFormer + SAMEO | **54.3** | **74.0** | **79.8** | **92.7** |
| RTMDet* + SAMEO | **55.3** | **75.2** | 72.7 | 85.8 |
| ConvNeXt-V2* + SAMEO | 54.1 | 73.1 | **80.8** | **94.0** |

### Zero-shot Performance Comparison

| Method | COCOA-cls AP | D2SA AP |
|------|-------------|---------|
| AISFormer (supervised) | 40.6 | 66.3 |
| RTMDet* + SAMEO† (zero-shot) | **54.4** | 68.4 |
| CO-DETR* + SAMEO† (zero-shot) | 54.0 | **75.0** |

### Ablation Study

| Ablation | AP | AP50 | AP75 |
|--------|-----|------|------|
| w/o IoU prediction | 52.4 | 73.2 | 57.8 |
| w/ IoU prediction | **54.3** | **74.0** | **59.7** |
| Amodal prompt only | 53.0 | 72.9 | 58.0 |
| Modal prompt only | 53.7 | 73.3 | 59.3 |
| Random prompt | **54.2** | **73.5** | **59.5** |

### Key Findings

- SAMEO's zero-shot performance outperforms the supervised AISFormer method, with an AP improvement of up to 13.8 points on COCOA-cls.
- Training with randomly chosen modal/amodal prompts yields the best performance and demonstrates the strongest generalization.
- Training only on occluded data leads to over-prediction; the dual-labeling mechanism effectively mitigates this issue.

## Highlights & Insights

1. **Excellent Decoupled Design Concept**: Decoupling amodal segmentation into "front-end detection + SAMEO decoding" enables a plug-and-play fashion, allowing any detector to be upgraded for amodal segmentation.
2. **Data-Engineering-Driven Zero-shot Capability**: Outperforming supervised methods in a zero-shot manner through large-scale data collection and cleaning demonstrates the critical importance of data quality and scale.
3. **Dual-Labeling Mechanism**: Discovers the over-prediction issue associated with training solely on occluded data and proposes a simple yet effective solution.

## Limitations & Future Work

- The model itself does not perform detection and remains dependent on the quality of the front-end detector.
- The synthetic occlusions in Amodal-LVIS may not fully reflect the complex occlusion patterns of the real world.
- Future work could explore end-to-end amodal segmentation schemes, or extend the method to video amodal segmentation.

## Related Work & Insights

- **SAM/EfficientSAM**: Validates that foundation segmentation models can be adapted to new tasks by fine-tuning only the decoder.
- **AISFormer**: Represents the current SOTA amodal method, but lacks flexibility.
- **pix2gestalt**: Provides large-scale synthetic amodal data, but suffers from incomplete annotations.

## Rating

⭐⭐⭐⭐ — The method is simple and elegant with solid data engineering. Successfully adapting the foundation model to amodal segmentation and outperforming supervised methods in a zero-shot manner is the core contribution. The plug-and-play nature of the framework holds high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAM4D: Segment Anything in Camera and LiDAR Streams](../../ICCV2025/autonomous_driving/sam4d_segment_anything_in_camera_and_lidar_streams.md)
- [\[CVPR 2025\] Cubify Anything: Scaling Indoor 3D Object Detection](cubify_anything_scaling_indoor_3d_object_detection.md)
- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[ICCV 2025\] Detect Anything 3D in the Wild](../../ICCV2025/autonomous_driving/detect_anything_3d_in_the_wild.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)

</div>

<!-- RELATED:END -->
